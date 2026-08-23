from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import io
import json
import os
import random
import math
import string

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .config import GeneratorConfig
from .drawing import (PALETTES, SYMBOLS, WORDS, degrade, draw_polyline, font, mask_bbox,
                      random_text, transform_pair)

BACKGROUND_COLORS = {
    "white": (255,255,255),
    "warm": (250,247,239),
    "cool": (242,248,252),
    "dark": (30,33,38),
    "paper": (238,232,213),
}


@dataclass(slots=True)
class CurveSpec:
    family: str
    degree: int
    coefficients: list[float]
    parameters: dict
    color: str
    width: int
    style: str
    label: str
    relation_group: int | None = None


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_save_image(image: Image.Image, path: Path, image_format: str, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format=image_format, **kwargs)
    temporary.replace(path)


class DatasetGenerator:
    def __init__(self, config: GeneratorConfig):
        config.validate()
        self.cfg = config

    def _curve_specs(self, rng: random.Random, np_rng: np.random.Generator, scale: int,
                     max_curves: int | None = None) -> list[CurveSpec]:
        upper = min(self.cfg.max_curves, max_curves or self.cfg.max_curves)
        lower = min(self.cfg.min_curves, upper)
        if rng.random()<self.cfg.empty_plot_probability:
            count=0
        else:
            choices=list(range(lower,upper+1))
            weights=[max(1,round(100*np.exp(-.48*abs(value-3)))) for value in choices]
            count=rng.choices(choices,weights)[0]
        palette=list(rng.choices(PALETTES,[36,31,29,4])[0]); rng.shuffle(palette)
        roll=rng.random()
        if roll<self.cfg.same_color_probability:
            colors=[rng.choice(palette)]*count
        elif roll<self.cfg.same_color_probability+self.cfg.grouped_color_probability:
            groups=rng.randint(2,min(4,count)) if count>1 else 1
            colors=[palette[i%groups] for i in range(count)]
        else:
            colors=[palette[i%len(palette)] for i in range(count)]
        specs=[]
        for i in range(count):
            family="polynomial"
            degree=rng.randint(self.cfg.min_degree,self.cfg.max_degree)
            coefficients=[]
            parameters={}
            if rng.random()<self.cfg.non_polynomial_probability:
                family=rng.choices(
                    ["sine_mix","gaussian_peaks","lorentzian_peaks","spline","step","damped"],
                    [20,22,10,20,13,15],
                )[0]
                degree=0
                if family=="sine_mix":
                    component_max=max(1,round(1+2*self.cfg.curve_complexity))
                    parameters={"components":[
                        [rng.uniform(.15,1.0),rng.uniform(.5,1.8+3.7*self.cfg.curve_complexity),rng.uniform(-np.pi,np.pi)]
                        for _ in range(rng.randint(1,component_max))
                    ],"trend":rng.uniform(-.7,.7)}
                elif family in ("gaussian_peaks","lorentzian_peaks"):
                    peak_max=max(1,round(1+3*self.cfg.curve_complexity))
                    parameters={"peaks":[
                        [rng.uniform(-1.4,1.4),rng.uniform(-.85,.85),rng.uniform(.025,.32)]
                        for _ in range(rng.randint(1,peak_max))
                    ],"trend":rng.uniform(-.45,.45),"baseline":rng.uniform(-.3,.3)}
                elif family=="spline":
                    n=rng.randint(5,max(5,round(6+7*self.cfg.curve_complexity)))
                    parameters={"control_x":np.linspace(-1,1,n).round(6).tolist(),
                                "control_y":np_rng.normal(0,1,n).round(6).tolist()}
                elif family=="step":
                    step_max=max(1,round(1+3*self.cfg.curve_complexity))
                    parameters={"steps":[
                        [rng.uniform(-1.1,1.1),rng.uniform(-.8,.8),rng.uniform(.015,.12)]
                        for _ in range(rng.randint(1,step_max))
                    ],"trend":rng.uniform(-.5,.5)}
                else:
                    parameters={"amplitude":rng.uniform(.5,1.4),"frequency":rng.uniform(2,3+6*self.cfg.curve_complexity),
                                "phase":rng.uniform(-np.pi,np.pi),"decay":rng.uniform(.1,1.8),
                                "trend":rng.uniform(-.35,.35)}
            else:
                cheb=np_rng.normal(0,1,degree+1)/np.sqrt(np.arange(degree+1)+1); cheb[0]*=.4
                power=np.polynomial.chebyshev.cheb2poly(cheb)
                coefficients=[round(float(v),8) for v in power]
            specs.append(CurveSpec(family,degree,coefficients,parameters,colors[i],
                         rng.randint(1,4)*scale,rng.choices(["solid","dashed","dotted","dashdot"],[65,16,10,9])[0],
                         rng.choice([f"{random_text(rng,1,2)} {i+1}",f"f{i+1}({rng.choice(SYMBOLS[:4])})"])))
        # Related series must remain distinct. Exact duplicate targets are
        # impossible to identify from pixels and teach query-based models to emit
        # duplicate masks for the same visible line.
        if count >= 2 and rng.random() < self.cfg.related_curves_probability:
            group_size = rng.randint(2, min(5, count))
            related_indices = rng.sample(range(count), group_size)
            base = specs[related_indices[0]]
            for position, related_index in enumerate(related_indices):
                original = specs[related_index]
                coefficients = list(base.coefficients)
                parameters = deepcopy(base.parameters)
                if position:
                    if base.family == "polynomial":
                        coefficients = [
                            round(value + rng.gauss(0, .035 / (degree + 1)), 8)
                            for degree, value in enumerate(coefficients)
                        ]
                        coefficients[0] = round(coefficients[0] + rng.choice((-1, 1)) * rng.uniform(.06, .18), 8)
                    elif base.family == "sine_mix":
                        parameters["components"] = [
                            [amplitude * rng.uniform(.94, 1.06),
                             frequency * rng.uniform(.98, 1.02),
                             phase + rng.uniform(-.08, .08)]
                            for amplitude, frequency, phase in parameters["components"]
                        ]
                        parameters["trend"] += rng.uniform(-.08, .08)
                    elif base.family in ("gaussian_peaks", "lorentzian_peaks"):
                        parameters["peaks"] = [
                            [amplitude * rng.uniform(.94, 1.06), center + rng.uniform(-.035, .035),
                             width * rng.uniform(.94, 1.06)]
                            for amplitude, center, width in parameters["peaks"]
                        ]
                        parameters["baseline"] += rng.choice((-1, 1)) * rng.uniform(.05, .14)
                    elif base.family == "spline":
                        offset = rng.choice((-1, 1)) * rng.uniform(.08, .20)
                        parameters["control_y"] = [
                            round(value + offset + rng.uniform(-.035, .035), 6)
                            for value in parameters["control_y"]
                        ]
                    elif base.family == "step":
                        parameters["steps"] = [
                            [amplitude * rng.uniform(.94, 1.06), center + rng.uniform(-.035, .035), width]
                            for amplitude, center, width in parameters["steps"]
                        ]
                        parameters["trend"] += rng.uniform(-.08, .08)
                    else:
                        parameters["amplitude"] *= rng.uniform(.94, 1.06)
                        parameters["frequency"] *= rng.uniform(.98, 1.02)
                        parameters["phase"] += rng.uniform(-.08, .08)
                        parameters["trend"] += rng.uniform(-.08, .08)
                specs[related_index] = CurveSpec(
                    family=base.family,
                    degree=base.degree,
                    coefficients=coefficients,
                    parameters=parameters,
                    color=original.color,
                    width=original.width,
                    style=original.style,
                    label=original.label,
                    relation_group=1,
                )
        return specs

    @staticmethod
    def _curve_values(x: np.ndarray, spec: CurveSpec) -> np.ndarray:
        p=spec.parameters
        if spec.family=="polynomial":
            return np.polynomial.polynomial.polyval(x,spec.coefficients)
        if spec.family=="sine_mix":
            y=p["trend"]*x
            for amplitude,frequency,phase in p["components"]:
                y=y+amplitude*np.sin(np.pi*frequency*x+phase)
            return y
        if spec.family in ("gaussian_peaks","lorentzian_peaks"):
            y=np.full_like(x,p["baseline"])+p["trend"]*x
            for amplitude,center,width in p["peaks"]:
                z=(x-center)/width
                y=y+amplitude*(np.exp(-.5*z*z) if spec.family=="gaussian_peaks" else 1/(1+z*z))
            return y
        if spec.family=="spline":
            return np.interp(x,np.asarray(p["control_x"]),np.asarray(p["control_y"]))
        if spec.family=="step":
            y=p["trend"]*x
            for amplitude,center,width in p["steps"]:
                y=y+amplitude*np.tanh((x-center)/width)
            return y
        envelope=np.exp(-p["decay"]*(x+1)/2)
        return p["amplitude"]*envelope*np.sin(np.pi*p["frequency"]*x+p["phase"])+p["trend"]*x

    @staticmethod
    def _segments(xp: np.ndarray, yp: np.ndarray, rect: tuple[int,int,int,int]) -> list[list[tuple[float,float]]]:
        l,t,r,b=rect
        valid=np.isfinite(xp)&np.isfinite(yp)&(xp>=l-5)&(xp<=r+5)&(yp>=t-5)&(yp<=b+5)
        chunks=np.split(np.arange(len(xp)),np.flatnonzero(np.diff(valid.astype(np.int8))!=0)+1)
        return [[(float(xp[j]),float(yp[j])) for j in c] for c in chunks if len(c)>1 and valid[c].all()]

    @staticmethod
    def _dsc_event_values(x: np.ndarray, event: dict) -> np.ndarray:
        """Asymmetric generalized-Gaussian thermal event."""
        delta=x-event["center"]
        width=np.where(delta<0,event["width_left"],event["width_right"])
        return event["amplitude"]*np.exp(-.5*np.abs(delta/width)**event["shape"])

    @staticmethod
    def _document_text_line(rng: random.Random,text_font,max_width: int) -> str:
        vocabulary=(
            "thermal analysis sample temperature heating cooling transition phase crystal "
            "melting enthalpy baseline observed measurement curve calorimetry material flow "
            "method result figure table polymer compound experiment rate peak onset data "
            "respectively indicates compared prepared recorded shows energy structure mass "
            "value normalized decomposition formation study behaviour significant region"
        ).split()
        tokens=[]
        target=max(18,max_width*rng.uniform(.58,.98))
        while True:
            roll=rng.random()
            if roll<.13:
                token="".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2,9)))
            elif roll<.22:
                token=rng.choice([f"{rng.uniform(0,500):.1f}",f"{rng.randint(1,99)}%",
                                  f"T{rng.choice(['g','m','c'])}",f"ΔH{rng.randint(1,4)}"])
            else:
                token=rng.choice(vocabulary)
            if rng.random()<.08: token=token.capitalize()
            if rng.random()<.12: token+=rng.choice([",",".",";",":"])
            candidate=" ".join((*tokens,token))
            if tokens and text_font.getlength(candidate)>target:
                break
            tokens.append(token)
            if len(tokens)>=18:
                break
        return " ".join(tokens)

    def _dsc_trace_values(self,x: np.ndarray,trace: dict,np_rng: np.random.Generator) -> np.ndarray:
        baseline=trace["baseline"]
        y=baseline[0]+baseline[1]*(x-.5)+baseline[2]*(x-.5)**2
        for step in trace["steps"]:
            z=np.clip((x-step["center"])/step["width"],-40,40)
            y=y+step["amplitude"]/(1+np.exp(-z))
        for event in trace["events"]:
            y=y+self._dsc_event_values(x,event)
        # DSC noise is usually weak and correlated rather than pointwise white.
        knots=max(24,round(len(x)/80))
        noise=np_rng.normal(0,trace["noise"],knots)
        y=y+np.interp(x,np.linspace(0,1,knots),noise)
        return y

    def _dsc_template(self,rng: random.Random,dense: bool=False) -> dict:
        if dense:
            event_count=rng.randint(7,self.cfg.dsc_max_events)
        else:
            event_count=rng.choices([1,2,3,4,5,6],[30,28,19,12,7,4])[0]
        polarity=rng.choice((-1,1))
        centers=[]
        while len(centers)<event_count:
            if centers and rng.random()<.38:
                center=rng.choice(centers)+rng.uniform(-.035,.035)
            else:
                center=rng.uniform(.08,.94)
            centers.append(min(.97,max(.03,center)))
        events=[]
        for center in centers:
            sharp=rng.random()<(.34 if not dense else .58)
            base_width=10**rng.uniform(-2.75,-1.72) if sharp else 10**rng.uniform(-1.70,-.68)
            asymmetry=10**rng.uniform(-.48,.48)
            amplitude=polarity*10**rng.uniform(-.22,.52)
            if rng.random()<.12:
                amplitude*=-rng.uniform(.25,.75)
            events.append({
                "center":center,
                "amplitude":amplitude,
                "width_left":base_width/math.sqrt(asymmetry),
                "width_right":base_width*math.sqrt(asymmetry),
                "shape":rng.uniform(1.25,3.5),
            })
        steps=[]
        for _ in range(rng.choices([0,1,2],[58,35,7])[0]):
            steps.append({
                "center":rng.uniform(.10,.90),
                "amplitude":rng.uniform(-.45,.45),
                "width":10**rng.uniform(-2.25,-1.05),
            })
        return {"polarity":polarity,"events":events,"steps":steps}

    @staticmethod
    def _mutate_dsc_template(template: dict,rng: random.Random,trace_index: int) -> dict:
        events=[]
        for event in template["events"]:
            if trace_index and rng.random()<.10:
                continue
            events.append({
                **event,
                "center":min(.98,max(.02,event["center"]+rng.gauss(0,.012))),
                "amplitude":event["amplitude"]*rng.uniform(.62,1.38),
                "width_left":event["width_left"]*rng.uniform(.72,1.42),
                "width_right":event["width_right"]*rng.uniform(.72,1.42),
            })
        if trace_index and rng.random()<.22:
            sign=template["polarity"]
            width=10**rng.uniform(-2.6,-.8)
            events.append({"center":rng.uniform(.08,.94),"amplitude":sign*10**rng.uniform(-.3,.35),
                           "width_left":width*rng.uniform(.65,1.2),
                           "width_right":width*rng.uniform(.8,1.6),"shape":rng.uniform(1.3,3.2)})
        if not events and template["events"]:
            event=template["events"][0]
            events=[{**event,"center":min(.98,max(.02,event["center"]+rng.gauss(0,.01)))}]
        steps=[{**step,"center":min(.97,max(.03,step["center"]+rng.gauss(0,.012))),
                "amplitude":step["amplitude"]*rng.uniform(.7,1.3)} for step in template["steps"]]
        return {"events":events,"steps":steps}

    def _render_dsc(self,rng: random.Random,np_rng: np.random.Generator,
                    forced_curve_max: int|None=None,force_plot_full: bool=False,
                    canvas_size: tuple[int,int]|None=None,forced_bg_kind: str|None=None):
        s=self.cfg.supersample
        w,h=canvas_size or (self.cfg.width*s,self.cfg.height*s)
        bg_kind=forced_bg_kind or rng.choices(["white","warm","cool","paper"],[76,10,8,6])[0]
        bg=BACKGROUND_COLORS[bg_kind]; fg=(24,24,24)
        image=Image.new("RGB",(w,h),bg); draw=ImageDraw.Draw(image)
        occluder=Image.new("L",(w,h),0); odraw=ImageDraw.Draw(occluder)
        page_layout=not force_plot_full and rng.random()<self.cfg.dsc_page_layout_probability
        page_elements=[]; hard_negative_count=0; caption=None
        if page_layout:
            min_fraction=self.cfg.dsc_plot_min_fraction
            outer_w=min(w-24*s,max(120*s,int(w*rng.uniform(min_fraction,.74))))
            outer_h=min(h-32*s,max(105*s,int(h*rng.uniform(min_fraction,.72))))
            outer_x=rng.randint(12*s,max(12*s,w-outer_w-12*s))
            outer_y=rng.randint(20*s,max(20*s,h-outer_h-12*s))
            outer=(outer_x,outer_y,outer_x+outer_w,outer_y+outer_h)
            margin_left=min(72*s,max(35*s,int(.16*outer_w)))
            margin_right=min(28*s,max(10*s,int(.05*outer_w)))
            margin_top=min(42*s,max(18*s,int(.10*outer_h)))
            margin_bottom=min(66*s,max(34*s,int(.17*outer_h)))
            left=outer_x+margin_left; right=outer_x+outer_w-margin_right
            top=outer_y+margin_top; bottom=outer_y+outer_h-margin_bottom
            article_font=font(rng.randint(7,11)*s)
            if rng.random()<self.cfg.dsc_surrounding_text_probability:
                header=random_text(rng,4,8)
                draw.text((12*s,5*s),header,fill=fg,font=font(rng.randint(10,15)*s))
                draw.text((w-rng.randint(48,85)*s,7*s),str(rng.randint(1,24)),fill=fg,font=article_font)
                page_elements.append({"type":"article_header"}); hard_negative_count+=1
                bands=[]
                if outer_y>42*s: bands.append((12*s,27*s,w-12*s,outer_y-7*s))
                if outer_y+outer_h<h-42*s: bands.append((12*s,outer_y+outer_h+9*s,w-12*s,h-10*s))
                if outer_x>125*s: bands.append((12*s,outer_y,outer_x-9*s,outer_y+outer_h))
                if outer_x+outer_w<w-125*s: bands.append((outer_x+outer_w+9*s,outer_y,w-12*s,outer_y+outer_h))
                for x0,y0,x1,y1 in bands:
                    spacing=rng.randint(9,14)*s
                    for yy in range(int(y0),int(y1),spacing):
                        if x1-x0>35*s:
                            indent=rng.choice([0,0,0,rng.randint(4,16)*s])
                            line=self._document_text_line(rng,article_font,int(x1-x0-indent))
                            shade=rng.choice([(55,55,55),(75,75,75),(95,95,95)])
                            draw.text((x0+indent,yy),line,fill=shade,font=article_font)
                    page_elements.append({"type":"surrounding_text","bbox":[x0,y0,x1,y1]})
                    hard_negative_count+=1
            if rng.random()<self.cfg.dsc_caption_probability:
                if outer_y+outer_h<h-24*s:
                    caption_xy=(outer_x,min(h-15*s,outer_y+outer_h+3*s))
                else:
                    caption_xy=(outer_x,max(1*s,outer_y-15*s))
                caption=f"Fig. {rng.randint(1,12)}. {random_text(rng,4,8)}"
                draw.text(caption_xy,caption,fill=fg,font=article_font)
                page_elements.append({"type":"caption","text":caption}); hard_negative_count+=1
            if rng.random()<self.cfg.dsc_foreign_graphics_probability:
                candidates=[]
                if outer_x>145*s: candidates.append((14*s,outer_y+12*s,outer_x-14*s,outer_y+outer_h-12*s))
                if outer_x+outer_w<w-145*s: candidates.append((outer_x+outer_w+14*s,outer_y+12*s,w-14*s,outer_y+outer_h-12*s))
                if outer_y>125*s: candidates.append((outer_x+10*s,25*s,outer_x+outer_w-10*s,outer_y-12*s))
                if outer_y+outer_h<h-125*s: candidates.append((outer_x+10*s,outer_y+outer_h+18*s,outer_x+outer_w-10*s,h-12*s))
                if candidates:
                    fx0,fy0,fx1,fy1=rng.choice(candidates)
                    draw.rectangle((fx0,fy0,fx1,fy1),outline=(125,125,125),width=max(1,s))
                    if rng.random()<.5:
                        for col in range(1,4):
                            xx=fx0+(fx1-fx0)*col/4; draw.line((xx,fy0,xx,fy1),fill=(170,170,170),width=max(1,s))
                        for row in range(1,5):
                            yy=fy0+(fy1-fy0)*row/5; draw.line((fx0,yy,fx1,yy),fill=(170,170,170),width=max(1,s))
                        kind="table"
                    else:
                        xs=np.linspace(fx0+4*s,fx1-4*s,70)
                        phase=rng.uniform(-math.pi,math.pi)
                        ys=(fy0+fy1)/2+.22*(fy1-fy0)*np.sin(np.linspace(phase,phase+rng.uniform(5,12),70))
                        draw.line([(float(xx),float(yy)) for xx,yy in zip(xs,ys)],fill=(90,90,90),width=max(1,s))
                        kind="foreign_plot"
                    page_elements.append({"type":kind,"bbox":[fx0,fy0,fx1,fy1]}); hard_negative_count+=1
        else:
            outer=(0,0,w,h)
            left_margin=min(rng.randint(68,105)*s,max(38*s,int(.22*w)))
            right_margin=min(rng.randint(20,48)*s,max(12*s,int(.10*w)))
            top_margin=min(rng.randint(34,70)*s,max(22*s,int(.18*h)))
            bottom_margin=min(rng.randint(58,88)*s,max(38*s,int(.25*h)))
            left=left_margin; right=w-right_margin
            top=top_margin; bottom=h-bottom_margin
        plot=(left,top,right,bottom)
        layout=rng.choices(["stacked","overlay"],[58,42])[0]
        upper=min(self.cfg.max_curves,forced_curve_max or self.cfg.max_curves,10)
        count=rng.choices(list(range(self.cfg.min_curves,upper+1)),
                          [max(1,round(100*math.exp(-.48*abs(v-3)))) for v in range(self.cfg.min_curves,upper+1)])[0]
        dense=rng.random()<self.cfg.dsc_dense_probability
        related=count>1 and rng.random()<self.cfg.related_curves_probability
        template=self._dsc_template(rng,dense=dense)
        traces=[]
        for i in range(count):
            individual_dense=dense and (i==0 or rng.random()<.45)
            shape=self._mutate_dsc_template(template,rng,i) if related else self._mutate_dsc_template(
                self._dsc_template(rng,dense=individual_dense),rng,0)
            shape["baseline"]=[rng.uniform(-.08,.08),rng.uniform(-.18,.18),rng.uniform(-.12,.12)]
            shape["noise"]=rng.uniform(.001,.012)
            traces.append(shape)
        # At least four samples per supersampled horizontal pixel, plus explicit
        # dense support around ultra-narrow events so needle peaks cannot vanish.
        samples=max(self.cfg.max_points,4*max(1,right-left))
        base_x=np.linspace(0,1,samples)
        extra=[]
        for trace in traces:
            for event in trace["events"]:
                width=min(event["width_left"],event["width_right"])
                extra.extend(event["center"]+width*np.linspace(-5,5,81))
        x=np.unique(np.clip(np.concatenate((base_x,np.asarray(extra))),0,1)) if extra else base_x
        values=[self._dsc_trace_values(x,trace,np_rng) for trace in traces]
        deviations=[]
        for y,trace in zip(values,traces):
            baseline=trace["baseline"][0]+trace["baseline"][1]*(x-.5)+trace["baseline"][2]*(x-.5)**2
            deviations.append(y-baseline)
        global_extent=max(.05,max(float(np.max(np.abs(value))) for value in deviations))
        if layout=="stacked":
            band=(bottom-top)/(count+.35)
            scale=.72*band/global_extent
            baselines=[top+(i+.62)*band for i in range(count)]
        else:
            scale=.42*(bottom-top)/global_extent
            baselines=[(top+bottom)/2+rng.uniform(-.04,.04)*(bottom-top) for _ in range(count)]
        axis_style=rng.choices(["open","box"],[72,28])[0]
        grid_style=rng.choices(["none","horizontal","major"],[72,17,11])[0]
        if grid_style!="none":
            grid=(220,220,220)
            for j in range(6):
                yy=top+(bottom-top)*j/5; draw.line((left,yy,right,yy),fill=grid,width=max(1,s))
            if grid_style=="major":
                for j in range(6):
                    xx=left+(right-left)*j/5; draw.line((xx,top,xx,bottom),fill=grid,width=max(1,s))
        aw=rng.randint(1,2)*s
        if axis_style=="box": draw.rectangle(plot,outline=fg,width=aw)
        else: draw.line((left,top,left,bottom,right,bottom),fill=fg,width=aw)
        temp_lo=rng.choice([20,25,30,40,50,60,80]); temp_hi=rng.choice([150,180,200,250,300,350,450])
        if temp_hi<=temp_lo+80: temp_hi=temp_lo+100
        tick_font=font(rng.randint(8,12)*s)
        for j in range(6):
            xx=left+(right-left)*j/5
            draw.line((xx,bottom,xx,bottom+5*s),fill=fg,width=max(1,s))
            label=f"{temp_lo+(temp_hi-temp_lo)*j/5:.0f}"
            draw.text((xx-10*s,bottom+7*s),label,fill=fg,font=tick_font)
        if layout=="overlay":
            for j in range(5):
                yy=top+(bottom-top)*j/4; draw.line((left-5*s,yy,left,yy),fill=fg,width=max(1,s))
        xlabel="Temperature (°C)"
        ylabel=rng.choice(["Heat Flow (mW)","Heat flow (W/g)","DSC (mW/mg)","Normalized Heat Flow"])
        label_font=font(rng.randint(11,16)*s)
        xlabel_width=label_font.getlength(xlabel)
        draw.text(((left+right-xlabel_width)/2,min(h-18*s,bottom+33*s)),xlabel,fill=fg,font=label_font)
        text_box=label_font.getbbox(ylabel)
        label_tile=Image.new("RGBA",(text_box[2]-text_box[0]+8*s,text_box[3]-text_box[1]+8*s),(0,0,0,0))
        ImageDraw.Draw(label_tile).text((4*s-text_box[0],4*s-text_box[1]),ylabel,fill=(*fg,255),font=label_font)
        label_tile=label_tile.rotate(90,Image.Resampling.BICUBIC,expand=True)
        label_xy=(max(0,left-45*s-label_tile.width//2),max(0,(top+bottom-label_tile.height)//2))
        image.paste(label_tile,label_xy,label_tile)
        label_alpha=label_tile.getchannel("A").point(lambda value: 255 if value>8 else 0)
        occluder.paste(255,label_xy,label_alpha)
        palette=list(rng.choice(PALETTES if rng.random()<.62 else [PALETTES[-1]])); rng.shuffle(palette)
        if rng.random()<.28: palette=["#222222"]*count
        masks=[]; curve_meta=[]
        for i,(trace,y,baseline_y) in enumerate(zip(traces,values,baselines),1):
            color=palette[(i-1)%len(palette)]
            width_px=rng.choices([1,2,3],[58,34,8])[0]
            width=width_px*s
            xp=left+x*(right-left)
            centered=y-trace["baseline"][0]
            yp=baseline_y-scale*centered
            points=self._segments(xp,yp,plot)
            mask=Image.new("L",(w,h),0); md=ImageDraw.Draw(mask)
            draw_polyline(draw,points,color,width,"solid")
            draw_polyline(md,points,255,max(width,s),"solid")
            masks.append(mask)
            curve_meta.append({
                "id":i,"family":"dsc_trace","degree":0,"coefficients":[],
                "function_parameters":{"baseline":trace["baseline"],"events":trace["events"],
                                       "steps":trace["steps"],"noise":trace["noise"]},
                "color":color,"line_width_px":width_px,"line_style":"solid","marker":None,
                "label":rng.choice([chr(96+i),f"Sample {i}",f"Form {i}"]),
                "relation_group":1 if related else None,
            })
        occluders=[]
        legend=(count>1 and layout=="overlay" and right-left>=220*s
                and rng.random()<self.cfg.legend_probability)
        if legend:
            lw=145*s; lh=(18*count+10)*s; lx=right-lw-8*s; ly=bottom-lh-8*s
            box=(lx,ly,lx+lw,ly+lh); draw.rectangle(box,fill=bg,outline=fg,width=max(1,s)); odraw.rectangle(box,fill=255)
            for i,curve in enumerate(curve_meta):
                yy=ly+(12+17*i)*s
                draw.line((lx+7*s,yy,lx+31*s,yy),fill=curve["color"],width=curve["line_width_px"]*s)
                draw.text((lx+36*s,yy-6*s),curve["label"],fill=fg,font=tick_font)
                odraw.line((lx+7*s,yy,lx+31*s,yy),fill=255,width=max(s,curve["line_width_px"]*s))
                odraw.text((lx+36*s,yy-6*s),curve["label"],fill=255,font=tick_font)
            occluders.append({"type":"legend","bbox":list(box),"opaque_background":True})
        elif layout=="stacked":
            for i,(curve,yy) in enumerate(zip(curve_meta,baselines)):
                xy=(right-75*s,int(yy-13*s)); draw.text(xy,curve["label"],fill=curve["color"],font=tick_font)
                odraw.text(xy,curve["label"],fill=255,font=tick_font)
            occluders.append({"type":"direct_labels"})
        annotations=[]
        if rng.random()<self.cfg.annotations_probability:
            candidates=[(i,e) for i,t in enumerate(traces) for e in t["events"]]
            rng.shuffle(candidates)
            for trace_index,event in candidates[:rng.randint(1,min(4,len(candidates)))]:
                xx=left+event["center"]*(right-left); yy=baselines[trace_index]-scale*event["amplitude"]
                temperature=temp_lo+event["center"]*(temp_hi-temp_lo)
                text_value=f"{temperature:.1f} °C"; xy=(int(xx-25*s),int(max(top,yy-24*s)))
                draw.text(xy,text_value,fill=fg,font=tick_font); odraw.text(xy,text_value,fill=255,font=tick_font)
                annotations.append({"type":"peak_temperature","temperature":round(temperature,2)})
        title=rng.choice(["DSC thermograms","Differential scanning calorimetry"]) if rng.random()<self.cfg.title_probability else None
        if canvas_size is not None and w<300*s:
            title=None
        if title:
            title_width=label_font.getlength(title)
            if title_width<=w-12*s:
                title_xy=((w-title_width)/2,max(0,top-28*s)); draw.text(title_xy,title,fill=fg,font=label_font)
                odraw.text(title_xy,title,fill=255,font=label_font)
            else:
                title=None
        watermark=None
        if page_layout and rng.random()<self.cfg.dsc_watermark_probability:
            watermark=rng.choice(["PREPRINT","ACCEPTED MANUSCRIPT","DRAFT","RESEARCH COPY"])
            wm=Image.new("L",(w,h),0); wm_draw=ImageDraw.Draw(wm)
            wm_draw.text((rng.randint(0,max(0,w//3)),rng.randint(0,max(0,h-70*s))),watermark,
                         fill=110,font=font(rng.randint(28,50)*s))
            wm=wm.rotate(rng.choice([-35,0,35,90]),Image.Resampling.BICUBIC,expand=False,fillcolor=0)
            image.paste((205,205,205),mask=wm)
            binary_wm=wm.point(lambda value: 255 if value>8 else 0)
            occluder=ImageChops.lighter(occluder,binary_wm)
            occluders.append({"type":"watermark","text":watermark}); hard_negative_count+=1
        visible=ImageChops.invert(occluder)
        masks=[ImageChops.multiply(mask,visible) for mask in masks]
        return image,masks,{
            "background":bg_kind,"axis_style":axis_style,"grid_style":grid_style,"title":title,
            "xlabel":xlabel,"ylabel":ylabel,"legend":legend,"occluders":occluders,
            "watermark":watermark,"page_layout":page_layout,"multi_panel":False,"panels":[],
            "hard_negative_count":hard_negative_count,"mask_semantics":"visible curve pixels only","curves":curve_meta,
            "plot_domain":"dsc","dsc_layout":layout,"dsc_polarity":template["polarity"],
            "dsc_dense_events":dense,"dsc_related_series":related,"annotations":annotations,
            "plot_bbox":list(plot),"figure_bbox":list(outer),"page_elements":page_elements,
            "caption":caption,
        }

    def _render_dsc_multipanel(self,rng: random.Random,np_rng: np.random.Generator,
                               canvas_size: tuple[int,int]|None=None,
                               forced_bg_kind: str|None=None,
                               forced_panel_count: int|None=None):
        """Render DSC panels directly at their final size instead of shrinking full plots."""
        s=self.cfg.supersample
        w,h=canvas_size or (self.cfg.width*s,self.cfg.height*s)
        maximum=min(self.cfg.max_panels,6)
        count=forced_panel_count or rng.choices(
            list(range(2,maximum+1)),[42,30,17,7,4][:maximum-1]
        )[0]
        margin=rng.randint(8,16)*s; gap=rng.randint(10,22)*s
        boxes=[]
        if count==2:
            layout=rng.choices(["top_bottom","left_right","staggered"],[46,34,20])[0]
            if layout=="top_bottom":
                cell_h=(h-2*margin-gap)//2
                panel_w=int((w-2*margin)*rng.uniform(.78,1.0)); x=(w-panel_w)//2
                boxes=[(x,margin,x+panel_w,margin+cell_h),
                       (x,margin+cell_h+gap,x+panel_w,h-margin)]
            elif layout=="left_right":
                cell_w=(w-2*margin-gap)//2
                panel_h=int((h-2*margin)*rng.uniform(.76,1.0)); y=(h-panel_h)//2
                boxes=[(margin,y,margin+cell_w,y+panel_h),
                       (margin+cell_w+gap,y,w-margin,y+panel_h)]
            else:
                panel_w=int((w-2*margin)*.72); panel_h=(h-2*margin-gap)//2
                boxes=[(margin,margin,margin+panel_w,margin+panel_h),
                       (w-margin-panel_w,margin+panel_h+gap,w-margin,h-margin)]
        elif count==3:
            available_width=w/s; available_height=h/s
            layouts=[]
            if available_width>=500 and available_height>=300:
                layouts.extend(["top_focus","bottom_focus"])
            if available_height>=390:
                layouts.append("three_rows")
            if not layouts:
                # The caller should normally allocate enough space. This safe
                # fallback favours width over creating unreadably short rows.
                layouts=["top_focus","bottom_focus"]
            layout=rng.choice(layouts)
            if layout=="three_rows":
                cell_h=(h-2*margin-2*gap)//3
                panel_w=int((w-2*margin)*rng.uniform(.80,1.0)); x=(w-panel_w)//2
                boxes=[(x,margin+i*(cell_h+gap),x+panel_w,margin+i*(cell_h+gap)+cell_h)
                       for i in range(3)]
            else:
                large_h=int((h-2*margin-gap)*.54); small_h=h-2*margin-gap-large_h
                half_w=(w-2*margin-gap)//2
                top_boxes=[(margin,margin,margin+half_w,margin+small_h),
                           (margin+half_w+gap,margin,w-margin,margin+small_h)]
                full_box=(margin,margin+small_h+gap,w-margin,h-margin)
                if layout=="top_focus":
                    full_box=(margin,margin,w-margin,margin+large_h)
                    y=margin+large_h+gap
                    top_boxes=[(margin,y,margin+half_w,h-margin),(margin+half_w+gap,y,w-margin,h-margin)]
                    boxes=[full_box,*top_boxes]
                else:
                    boxes=[*top_boxes,full_box]
        else:
            layout="grid"
            columns=2 if count<=4 else 3
            rows=math.ceil(count/columns)
            cell_w=(w-2*margin-gap*(columns-1))//columns
            cell_h=(h-2*margin-gap*(rows-1))//rows
            for index in range(count):
                row=index//columns; column=index%columns
                x0=margin+column*(cell_w+gap); y0=margin+row*(cell_h+gap)
                boxes.append((x0,y0,x0+cell_w,y0+cell_h))
        bg_kind=forced_bg_kind or rng.choices(["white","warm","cool","paper"],[78,10,7,5])[0]
        bg=BACKGROUND_COLORS[bg_kind]; fg=(24,24,24)
        image=Image.new("RGB",(w,h),bg); masks=[]; curves=[]; panels=[]
        panel_occluder=Image.new("L",(w,h),0); panel_odraw=ImageDraw.Draw(panel_occluder)
        for panel_index,(x0,y0,x1,y1) in enumerate(boxes,1):
            panel_image,panel_masks,panel_meta=self._render_dsc(
                rng,np_rng,
                forced_curve_max=min(self.cfg.max_curves,self.cfg.max_curves_per_panel),
                force_plot_full=True,canvas_size=(x1-x0,y1-y0),forced_bg_kind=bg_kind,
            )
            image.paste(panel_image,(x0,y0))
            panel_curve_ids=[]
            for panel_mask,panel_curve in zip(panel_masks,panel_meta["curves"]):
                placed=Image.new("L",(w,h),0); placed.paste(panel_mask,(x0,y0))
                curve_id=len(curves)+1; masks.append(placed)
                curves.append({**panel_curve,"id":curve_id,"panel_id":panel_index})
                panel_curve_ids.append(curve_id)
            label=chr(ord("a")+panel_index-1); label_xy=(x0+3*s,y0+2*s)
            label_font=font(rng.randint(13,20)*s)
            ImageDraw.Draw(image).text(label_xy,label,fill=fg,font=label_font)
            panel_odraw.text(label_xy,label,fill=255,font=label_font)
            panels.append({
                "id":panel_index,"label":label,"base_bbox":[x0,y0,x1,y1],
                "rendered_size_px":[round((x1-x0)/s),round((y1-y0)/s)],
                "rendered_natively":True,"curve_ids":panel_curve_ids,
                "axis_style":panel_meta["axis_style"],"grid_style":panel_meta["grid_style"],
                "legend":panel_meta["legend"],"dsc_layout":panel_meta["dsc_layout"],
            })
        visible=ImageChops.invert(panel_occluder)
        masks=[ImageChops.multiply(mask,visible) for mask in masks]
        return image,masks,{
            "background":bg_kind,"axis_style":"multi_panel","grid_style":"mixed",
            "title":None,"xlabel":None,"ylabel":None,
            "legend":any(panel["legend"] for panel in panels),
            "occluders":[{"type":"panel_label"} for _ in panels],"watermark":None,
            "page_layout":True,"multi_panel":True,"panels":panels,"caption":None,
            "hard_negative_count":0,"mask_semantics":"visible curve pixels only","curves":curves,
            "plot_domain":"dsc","dsc_panel_layout":layout,
        }

    def _render_dsc_multipanel_page(self,rng: random.Random,np_rng: np.random.Generator):
        """Place a natively rendered multipanel DSC figure inside an article page."""
        s=self.cfg.supersample; w,h=self.cfg.width*s,self.cfg.height*s
        bg_kind=rng.choices(["white","warm","cool","paper"],[78,10,7,5])[0]
        bg=BACKGROUND_COLORS[bg_kind]; fg=(24,24,24)
        image=Image.new("RGB",(w,h),bg); draw=ImageDraw.Draw(image)
        maximum=min(self.cfg.max_panels,6)
        panel_count=rng.choices(
            list(range(2,maximum+1)),[42,30,17,7,4][:maximum-1]
        )[0]
        if panel_count==2:
            minimum_width_fraction=.58; minimum_height_fraction=.60
        elif panel_count==3:
            minimum_width_fraction=.70; minimum_height_fraction=.76
        else:
            minimum_width_fraction=.82; minimum_height_fraction=.80
        min_fraction=self.cfg.dsc_plot_min_fraction
        min_w=max(minimum_width_fraction,min_fraction)
        min_h=max(minimum_height_fraction,min_fraction)
        figure_w=min(w-28*s,max(300*s,int(w*rng.uniform(min_w,.88))))
        figure_h=min(h-42*s,max(280*s,int(h*rng.uniform(min_h,.86))))
        placement=rng.choice(["top","bottom","left","right","center"])
        if placement=="top":
            figure_x=(w-figure_w)//2; figure_y=rng.randint(22,42)*s
        elif placement=="bottom":
            figure_x=(w-figure_w)//2; figure_y=h-figure_h-rng.randint(18,36)*s
        elif placement=="left":
            figure_x=rng.randint(14,30)*s; figure_y=(h-figure_h)//2
        elif placement=="right":
            figure_x=w-figure_w-rng.randint(14,30)*s; figure_y=(h-figure_h)//2
        else:
            figure_x=(w-figure_w)//2; figure_y=(h-figure_h)//2
        figure=(figure_x,figure_y,figure_x+figure_w,figure_y+figure_h)
        page_elements=[]; hard_negative_count=0
        article_font=font(rng.randint(7,11)*s)
        if rng.random()<self.cfg.dsc_surrounding_text_probability:
            header=random_text(rng,4,8)
            draw.text((12*s,5*s),header,fill=fg,font=font(rng.randint(10,15)*s))
            draw.text((w-rng.randint(48,85)*s,7*s),str(rng.randint(1,24)),fill=fg,font=article_font)
            page_elements.append({"type":"article_header"}); hard_negative_count+=1
            fx0,fy0,fx1,fy1=figure
            bands=[]
            if fy0>42*s: bands.append((12*s,27*s,w-12*s,fy0-7*s))
            if fy1<h-42*s: bands.append((12*s,fy1+9*s,w-12*s,h-10*s))
            if fx0>125*s: bands.append((12*s,fy0,fx0-9*s,fy1))
            if fx1<w-125*s: bands.append((fx1+9*s,fy0,w-12*s,fy1))
            for x0,y0,x1,y1 in bands:
                spacing=rng.randint(9,14)*s
                for yy in range(int(y0),int(y1),spacing):
                    if x1-x0>35*s:
                        indent=rng.choice([0,0,0,rng.randint(4,16)*s])
                        line=self._document_text_line(rng,article_font,int(x1-x0-indent))
                        shade=rng.choice([(55,55,55),(75,75,75),(95,95,95)])
                        draw.text((x0+indent,yy),line,fill=shade,font=article_font)
                page_elements.append({"type":"surrounding_text","bbox":[x0,y0,x1,y1]})
                hard_negative_count+=1
        caption=None
        if rng.random()<self.cfg.dsc_caption_probability:
            if figure[3]<h-24*s:
                caption_xy=(figure[0],min(h-15*s,figure[3]+3*s))
            else:
                caption_xy=(figure[0],max(1*s,figure[1]-15*s))
            caption=f"Fig. {rng.randint(1,12)}. {random_text(rng,4,8)}"
            draw.text(caption_xy,caption,fill=fg,font=article_font)
            page_elements.append({"type":"caption","text":caption}); hard_negative_count+=1
        if rng.random()<self.cfg.dsc_foreign_graphics_probability:
            fx0,fy0,fx1,fy1=figure; candidates=[]
            if fx0>150*s: candidates.append((14*s,fy0+10*s,fx0-14*s,fy1-10*s))
            if fx1<w-150*s: candidates.append((fx1+14*s,fy0+10*s,w-14*s,fy1-10*s))
            if fy0>130*s: candidates.append((fx0+10*s,25*s,fx1-10*s,fy0-12*s))
            if fy1<h-130*s: candidates.append((fx0+10*s,fy1+18*s,fx1-10*s,h-12*s))
            if candidates:
                x0,y0,x1,y1=rng.choice(candidates)
                draw.rectangle((x0,y0,x1,y1),outline=(125,125,125),width=max(1,s))
                if rng.random()<.5:
                    for col in range(1,4):
                        xx=x0+(x1-x0)*col/4; draw.line((xx,y0,xx,y1),fill=(170,170,170),width=max(1,s))
                    for row in range(1,5):
                        yy=y0+(y1-y0)*row/5; draw.line((x0,yy,x1,yy),fill=(170,170,170),width=max(1,s))
                    kind="table"
                else:
                    xs=np.linspace(x0+4*s,x1-4*s,70); phase=rng.uniform(-math.pi,math.pi)
                    ys=(y0+y1)/2+.22*(y1-y0)*np.sin(np.linspace(phase,phase+rng.uniform(5,12),70))
                    draw.line([(float(xx),float(yy)) for xx,yy in zip(xs,ys)],fill=(90,90,90),width=max(1,s))
                    kind="foreign_plot"
                page_elements.append({"type":kind,"bbox":[x0,y0,x1,y1]}); hard_negative_count+=1
        figure_image,figure_masks,meta=self._render_dsc_multipanel(
            rng,np_rng,canvas_size=(figure_w,figure_h),forced_bg_kind=bg_kind,
            forced_panel_count=panel_count,
        )
        image.paste(figure_image,(figure_x,figure_y))
        masks=[]
        for mask in figure_masks:
            placed=Image.new("L",(w,h),0); placed.paste(mask,(figure_x,figure_y)); masks.append(placed)
        for panel in meta["panels"]:
            x0,y0,x1,y1=panel["base_bbox"]
            panel["base_bbox"]=[x0+figure_x,y0+figure_y,x1+figure_x,y1+figure_y]
        watermark=None
        if rng.random()<self.cfg.dsc_watermark_probability:
            watermark=rng.choice(["PREPRINT","ACCEPTED MANUSCRIPT","DRAFT","RESEARCH COPY"])
            wm=Image.new("L",(w,h),0); wm_draw=ImageDraw.Draw(wm)
            wm_draw.text((rng.randint(0,max(0,w//3)),rng.randint(0,max(0,h-70*s))),watermark,
                         fill=110,font=font(rng.randint(28,50)*s))
            wm=wm.rotate(rng.choice([-35,0,35,90]),Image.Resampling.BICUBIC,expand=False,fillcolor=0)
            image.paste((205,205,205),mask=wm)
            visible=ImageChops.invert(wm.point(lambda value: 255 if value>8 else 0))
            masks=[ImageChops.multiply(mask,visible) for mask in masks]
            meta["occluders"].append({"type":"watermark","text":watermark}); hard_negative_count+=1
        return image,masks,{
            **meta,"background":bg_kind,"page_layout":True,"watermark":watermark,
            "caption":caption,"hard_negative_count":hard_negative_count,
            "figure_bbox":list(figure),"page_elements":page_elements,
            "dsc_document_layout":True,"dsc_document_placement":placement,
        }

    def _render_multipanel(self,rng: random.Random,np_rng: np.random.Generator):
        s=self.cfg.supersample; w,h=self.cfg.width*s,self.cfg.height*s
        panel_options=list(range(2,self.cfg.max_panels+1))
        panel_weights=[{2:18,3:30,4:30,5:14,6:8}[value] for value in panel_options]
        count=rng.choices(panel_options,panel_weights)[0]
        if count==2:
            row_counts=rng.choice([[2],[1,1]])
        elif count==3:
            row_counts=rng.choice([[3],[2,1],[1,2]])
        elif count==4:
            row_counts=rng.choice([[2,2],[4]])
        elif count==5:
            row_counts=rng.choice([[3,2],[2,3]])
        else:
            row_counts=rng.choice([[3,3],[2,2,2]])
        bg_kind=rng.choice(["white","warm","cool","paper"])
        bg=BACKGROUND_COLORS[bg_kind]; fg=(25,25,25)
        image=Image.new("RGB",(w,h),bg)
        masks=[]; curves=[]; panels=[]; global_occluder=Image.new("L",(w,h),0)
        global_occluder_draw=ImageDraw.Draw(global_occluder)
        outer_x=rng.randint(6,18)*s; outer_y=rng.randint(8,22)*s
        gap_x=rng.randint(8,22)*s; gap_y=rng.randint(12,28)*s
        caption_h=(rng.randint(18,34)*s if rng.random()<.55 else 0)
        available_h=h-2*outer_y-caption_h-gap_y*(len(row_counts)-1)
        row_h=max(16*s,available_h//len(row_counts))
        panel_index=0
        for row_index,row_count in enumerate(row_counts):
            available_w=w-2*outer_x-gap_x*(row_count-1)
            cell_w=max(16*s,available_w//row_count)
            y0=outer_y+row_index*(row_h+gap_y)
            for column_index in range(row_count):
                x0=outer_x+column_index*(cell_w+gap_x)
                x1=min(w-outer_x,x0+cell_w); y1=min(h-caption_h-outer_y,y0+row_h)
                panel_image,panel_masks,panel_meta=self._render_base(
                    rng,np_rng,allow_multipanel=False,force_plot_full=True,
                    forced_bg_kind=bg_kind,
                    forced_curve_max=min(self.cfg.max_curves,self.cfg.max_curves_per_panel),
                )
                panel_image=panel_image.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
                image.paste(panel_image,(x0,y0))
                panel_curve_ids=[]
                for panel_mask,panel_curve in zip(panel_masks,panel_meta["curves"]):
                    resized=panel_mask.resize((x1-x0,y1-y0),Image.Resampling.NEAREST)
                    placed=Image.new("L",(w,h),0); placed.paste(resized,(x0,y0))
                    curve_id=len(curves)+1
                    masks.append(placed)
                    curves.append({**panel_curve,"id":curve_id,"panel_id":panel_index+1})
                    panel_curve_ids.append(curve_id)
                label=chr(ord("a")+panel_index)
                label_xy=(x0+2*s,max(0,y1-18*s))
                label_font=font(rng.randint(13,22)*s)
                ImageDraw.Draw(image).text(label_xy,label,fill=fg,font=label_font)
                global_occluder_draw.text(label_xy,label,fill=255,font=label_font)
                panels.append({"id":panel_index+1,"label":label,"base_bbox":[x0,y0,x1,y1],
                               "curve_ids":panel_curve_ids,"axis_style":panel_meta["axis_style"],
                               "grid_style":panel_meta["grid_style"],"legend":panel_meta["legend"]})
                panel_index+=1
        caption=None
        if caption_h:
            caption=random_text(rng,4,8)
            xy=(outer_x,h-caption_h+3*s); caption_font=font(rng.randint(9,14)*s)
            ImageDraw.Draw(image).text(xy,caption,fill=fg,font=caption_font)
            global_occluder_draw.text(xy,caption,fill=255,font=caption_font)
        visible=ImageChops.invert(global_occluder)
        masks=[ImageChops.multiply(mask,visible) for mask in masks]
        return image,masks,{
            "background":bg_kind,"axis_style":"multi_panel","grid_style":"mixed",
            "title":None,"xlabel":None,"ylabel":None,"legend":any(panel["legend"] for panel in panels),
            "occluders":[{"type":"panel_label"} for _ in panels],
            "watermark":None,"page_layout":True,"multi_panel":True,
            "panels":panels,"caption":caption,"hard_negative_count":0,
            "mask_semantics":"visible curve pixels only","curves":curves,
            "plot_domain":"dsc" if self.cfg.plot_domain=="dsc" else self.cfg.plot_domain,
        }

    def _render_base(self,rng: random.Random,np_rng: np.random.Generator,
                     allow_multipanel: bool=True,force_plot_full: bool=False,
                     forced_bg_kind: str|None=None,
                     forced_curve_max: int|None=None):
        use_dsc=self.cfg.plot_domain=="dsc" or (
            self.cfg.plot_domain=="mixed" and rng.random()<self.cfg.dsc_probability
        )
        if use_dsc:
            if allow_multipanel and rng.random()<self.cfg.multi_panel_probability:
                if rng.random()<self.cfg.dsc_multipanel_page_probability:
                    return self._render_dsc_multipanel_page(rng,np_rng)
                return self._render_dsc_multipanel(rng,np_rng)
            return self._render_dsc(rng,np_rng,forced_curve_max,force_plot_full,
                                    forced_bg_kind=forced_bg_kind)
        if allow_multipanel and rng.random()<self.cfg.multi_panel_probability:
            return self._render_multipanel(rng,np_rng)
        s=self.cfg.supersample; w,h=self.cfg.width*s,self.cfg.height*s
        pad=int(min(w,h)*rng.uniform(.05,.15))
        bg_kind=forced_bg_kind or rng.choice(list(BACKGROUND_COLORS))
        bg=BACKGROUND_COLORS[bg_kind]
        fg=(235,235,235) if bg_kind=="dark" else (25,25,25)
        image=Image.new("RGB",(w,h),bg); draw=ImageDraw.Draw(image)
        # Everything drawn into this layer after the curves is genuinely opaque
        # in the final image and must therefore be removed from every curve mask.
        occluder=Image.new("L",(w,h),0); occluder_draw=ImageDraw.Draw(occluder)
        occluder_meta=[]
        page_layout=not force_plot_full and rng.random()<self.cfg.page_layout_probability
        if page_layout:
            minimum=self.cfg.page_plot_min_fraction
            plot_w=int(w*rng.uniform(minimum,.90)); plot_h=int(h*rng.uniform(minimum,.82))
            l=rng.randint(int(.05*w),max(int(.05*w),w-plot_w-int(.05*w)))
            t=rng.randint(int(.08*h),max(int(.08*h),h-plot_h-int(.10*h)))
            plot=(l,t,l+plot_w,t+plot_h)
        else:
            plot=(pad+rng.randint(0,pad),pad+rng.randint(0,pad),w-pad-rng.randint(0,pad),h-pad-rng.randint(0,pad))
        l,t,r,b=plot
        layout_font=font(rng.randint(8,13)*s)
        if page_layout:
            header=random_text(rng,3,7)
            draw.text((rng.randint(8,30)*s,rng.randint(4,14)*s),header,fill=fg,font=layout_font)
            draw.text((w-rng.randint(45,85)*s,rng.randint(4,14)*s),str(rng.randint(1,24)),fill=fg,font=layout_font)
            if rng.random()<self.cfg.dense_text_probability:
                bands=[]
                if t>45*s: bands.append((8*s,30*s,w-8*s,max(31*s,t-8*s)))
                if b<h-35*s: bands.append((8*s,b+12*s,w-8*s,h-8*s))
                if l>100*s: bands.append((8*s,t+10*s,l-10*s,b-5*s))
                if r<w-100*s: bands.append((r+10*s,t+10*s,w-8*s,b-5*s))
                for x0,y0,x1,y1 in bands:
                    for yy in range(int(y0),int(y1),rng.randint(8,13)*s):
                        if x1-x0>20*s:
                            length=rng.uniform(.55,.98)*(x1-x0)
                            shade=(75,75,75) if bg_kind=="dark" else (115,115,115)
                            draw.line((x0,yy,x0+length,yy),fill=shade,width=max(1,s))
        if bg_kind=="paper":
            for _ in range(max(80,w*h//12000)):
                x,y=rng.randrange(w),rng.randrange(h); c=rng.randint(195,235)
                draw.point((x,y),fill=(c,c,max(0,min(255,rng.randint(c-8,c+8)))))
        axis_style=rng.choice(["box","open","cross","minimal","none","arrows"])
        grid_style=rng.choices(["none","major","minor","both","horizontal","vertical"],[24,27,12,15,11,11])[0]
        nx,ny=rng.randint(4,10),rng.randint(4,9); grid=(75,75,75) if bg_kind=="dark" else (195,195,195)
        if grid_style!="none":
            for i in range(nx+1):
                if grid_style!="horizontal":
                    x=l+(r-l)*i/nx; draw.line((x,t,x,b),fill=grid,width=max(1,s))
            for i in range(ny+1):
                if grid_style!="vertical":
                    y=t+(b-t)*i/ny; draw.line((l,y,r,y),fill=grid,width=max(1,s))
            if grid_style in ("minor","both"):
                for i in range(nx*2+1):
                    x=l+(r-l)*i/(nx*2); draw.line((x,t,x,b),fill=grid,width=1)
                for i in range(ny*2+1):
                    y=t+(b-t)*i/(ny*2); draw.line((l,y,r,y),fill=grid,width=1)
        hard_negative_count=0
        if rng.random()<self.cfg.hard_negatives_probability:
            hard_negative_count=rng.randint(1,6)
            for _ in range(hard_negative_count):
                kind=rng.choice(["reference_h","reference_v","errorbar","scatter","arrow"])
                shade=rng.choice([fg,grid,(110,110,110)])
                if kind=="reference_h":
                    yy=rng.randint(t,b); draw.line((l,yy,r,yy),fill=shade,width=max(1,s))
                elif kind=="reference_v":
                    xx=rng.randint(l,r); draw.line((xx,t,xx,b),fill=shade,width=max(1,s))
                elif kind=="scatter":
                    for _ in range(rng.randint(4,18)):
                        xx,yy=rng.randint(l,r),rng.randint(t,b); rr=rng.randint(1,3)*s
                        draw.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),fill=shade)
                elif kind=="errorbar":
                    xx,yy=rng.randint(l,r),rng.randint(t,b); span=rng.randint(5,22)*s
                    draw.line((xx,yy-span,xx,yy+span),fill=shade,width=max(1,s))
                    draw.line((xx-3*s,yy-span,xx+3*s,yy-span),fill=shade,width=max(1,s))
                    draw.line((xx-3*s,yy+span,xx+3*s,yy+span),fill=shade,width=max(1,s))
                else:
                    x0,y0=rng.randint(l,r),rng.randint(t,b); x1,y1=x0+rng.randint(-30,30)*s,y0+rng.randint(-30,30)*s
                    draw.line((x0,y0,x1,y1),fill=shade,width=max(1,s))
        aw=rng.randint(1,3)*s
        if axis_style=="box": draw.rectangle(plot,outline=fg,width=aw)
        elif axis_style in ("open","minimal"): draw.line((l,t,l,b,r,b),fill=fg,width=aw)
        elif axis_style=="cross":
            draw.line((l,(t+b)//2,r,(t+b)//2),fill=fg,width=aw); draw.line(((l+r)//2,t,(l+r)//2,b),fill=fg,width=aw)
        elif axis_style=="arrows":
            draw.line((l,b,r,b),fill=fg,width=aw); draw.line((l,b,l,t),fill=fg,width=aw)
            draw.polygon([(r,b),(r-9*s,b-4*s),(r-9*s,b+4*s)],fill=fg)
            draw.polygon([(l,t),(l-4*s,t+9*s),(l+4*s,t+9*s)],fill=fg)
        fsmall=font(rng.randint(8,13)*s)
        if axis_style!="none" and rng.random()<.86:
            xlo,xhi=rng.choice([(-1,1),(-5,5),(0,10),(0,100),(-10,30)])
            ylo,yhi=rng.choice([(-1,1),(-5,5),(0,10),(0,100),(-20,20)])
            for i in range(nx+1):
                x=l+(r-l)*i/nx; draw.line((x,b,x,b+4*s),fill=fg,width=s)
                if rng.random()>.08: draw.text((x-9*s,b+5*s),f"{xlo+(xhi-xlo)*i/nx:g}",fill=fg,font=fsmall)
            for i in range(ny+1):
                y=b-(b-t)*i/ny; draw.line((l-4*s,y,l,y),fill=fg,width=s)
                if rng.random()>.08: draw.text((max(0,l-34*s),y-6*s),f"{ylo+(yhi-ylo)*i/ny:g}",fill=fg,font=fsmall)
        specs=self._curve_specs(rng,np_rng,s,forced_curve_max)
        bg_luminance=sum(bg)/3
        contrast_colors=["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9"]
        for spec in specs:
            rgb=tuple(int(spec.color[index:index+2],16) for index in (1,3,5))
            if abs(sum(rgb)/3-bg_luminance)<65:
                spec.color=rng.choice(contrast_colors)
        masks=[Image.new("L",(w,h),0) for _ in specs]
        x=np.linspace(-1,1,self.cfg.max_points); curve_meta=[]
        relation_transforms: dict[int,tuple[float,float]]={}
        for idx,(spec,mask) in enumerate(zip(specs,masks),1):
            y=self._curve_values(x,spec)
            qlo,qhi=np.nanpercentile(y,[3,97]); span=max(.05,qhi-qlo)
            normalized=(y-(qlo+qhi)/2)/span
            if spec.relation_group is not None and spec.relation_group in relation_transforms:
                amplitude,offset=relation_transforms[spec.relation_group]
                amplitude*=rng.uniform(.96,1.04)
                offset+=rng.choice((-1,1))*rng.uniform(.035,.085)
            else:
                amplitude=rng.uniform(.55,.95)
                offset=rng.uniform(-.24,.24)
                if spec.relation_group is not None:
                    relation_transforms[spec.relation_group]=(amplitude,offset)
            yn=normalized*amplitude+offset
            xp=l+(x+1)*.5*(r-l); yp=(t+b)/2-yn*(b-t); segments=self._segments(xp,yp,plot)
            md=ImageDraw.Draw(mask); draw_polyline(draw,segments,spec.color,spec.width,spec.style)
            draw_polyline(md,segments,255,spec.width+max(1,s),spec.style)
            marker=None
            if rng.random()<self.cfg.markers_probability:
                marker=rng.choice(["circle","square","cross"]); step=rng.randint(35,90)
                for j in range(rng.randint(6,16),len(x),step):
                    px,py=float(xp[j]),float(yp[j]); rr=rng.randint(2,5)*s
                    if l<=px<=r and t<=py<=b:
                        if marker=="circle":
                            marker_width=max(1,s)
                            draw.ellipse((px-rr,py-rr,px+rr,py+rr),outline=spec.color,width=marker_width)
                            md.ellipse((px-rr,py-rr,px+rr,py+rr),outline=255,width=marker_width)
                        elif marker=="square":
                            marker_width=max(1,s)
                            draw.rectangle((px-rr,py-rr,px+rr,py+rr),outline=spec.color,width=marker_width)
                            md.rectangle((px-rr,py-rr,px+rr,py+rr),outline=255,width=marker_width)
                        else:
                            for yy in (-rr,rr):
                                draw.line((px-rr,py+yy,px+rr,py-yy),fill=spec.color,width=spec.width)
                                md.line((px-rr,py+yy,px+rr,py-yy),fill=255,width=spec.width+1)
            curve_meta.append({"id":idx,"family":spec.family,"degree":spec.degree,
                               "coefficients":spec.coefficients,"function_parameters":spec.parameters,"color":spec.color,
                               "line_width_px":spec.width/s,"line_style":spec.style,"marker":marker,
                               "label":spec.label,"relation_group":spec.relation_group})
        title=random_text(rng,2,6) if rng.random()<self.cfg.title_probability else None
        xlabel=rng.choice(SYMBOLS+WORDS) if rng.random()<self.cfg.labels_probability else None
        ylabel=rng.choice(SYMBOLS+WORDS) if rng.random()<self.cfg.labels_probability else None
        if title:
            xy=((l+r)//2-len(title)*4*s,max(0,t-27*s)); title_font=font(rng.randint(13,22)*s)
            draw.text(xy,title,fill=fg,font=title_font)
            occluder_draw.text(xy,title,fill=255,font=title_font)
        if xlabel:
            xy=((l+r)//2,min(h-18*s,b+25*s)); xlabel_font=font(rng.randint(9,15)*s)
            draw.text(xy,xlabel,fill=fg,font=xlabel_font)
            occluder_draw.text(xy,xlabel,fill=255,font=xlabel_font)
        if ylabel:
            xy=(max(0,l-28*s),max(0,t-18*s)); ylabel_font=font(rng.randint(9,15)*s)
            draw.text(xy,ylabel,fill=fg,font=ylabel_font)
            occluder_draw.text(xy,ylabel,fill=255,font=ylabel_font)
        legend=bool(specs) and rng.random()<self.cfg.legend_probability
        if legend:
            lw,lh=rng.randint(95,180)*s,(len(specs)*rng.randint(13,20)+12)*s
            lx=rng.choice([l+8*s,max(l,r-lw-8*s)]); ly=rng.choice([t+8*s,max(t,b-lh-8*s)])
            legend_box=(lx,ly,lx+lw,ly+lh); opaque_background=rng.random()<.8
            if opaque_background:
                draw.rectangle(legend_box,fill=bg,outline=fg,width=s)
                occluder_draw.rectangle(legend_box,fill=255)
            for i,spec in enumerate(specs):
                yy=ly+(10+i*16)*s; line=(lx+7*s,yy,lx+30*s,yy); text_xy=(lx+35*s,yy-6*s)
                draw.line(line,fill=spec.color,width=max(s,spec.width))
                draw.text(text_xy,spec.label,fill=fg,font=fsmall)
                # Legend samples and text are not plot-curve pixels either.
                occluder_draw.line(line,fill=255,width=max(s,spec.width))
                occluder_draw.text(text_xy,spec.label,fill=255,font=fsmall)
            occluder_meta.append({"type":"legend","bbox":list(legend_box),
                                  "opaque_background":opaque_background})
        if rng.random()<self.cfg.annotations_probability:
            for _ in range(rng.randint(1,4)):
                ax,ay=rng.randint(l,r),rng.randint(t,b); annotation=random_text(rng,1,3)
                line=(ax,ay+12*s,ax+rng.randint(-35,35)*s,ay+rng.randint(15,45)*s)
                draw.text((ax,ay),annotation,fill=fg,font=fsmall); draw.line(line,fill=fg,width=s)
                occluder_draw.text((ax,ay),annotation,fill=255,font=fsmall)
                occluder_draw.line(line,fill=255,width=s)
                occluder_meta.append({"type":"annotation"})
        if rng.random()<self.cfg.occlusion_probability:
            for _ in range(rng.randint(1,3)):
                x0,y0=rng.randint(0,w-20*s),rng.randint(0,h-20*s); ww,hh=rng.randint(10,80)*s,rng.randint(8,55)*s
                box=(x0,y0,min(w,x0+ww),min(h,y0+hh))
                draw.rectangle(box,fill=rng.choice([bg,(255,255,255),(210,210,210)]))
                occluder_draw.rectangle(box,fill=255)
                occluder_meta.append({"type":"rectangle","bbox":list(box)})
        watermark=None
        if not force_plot_full and rng.random()<self.cfg.watermark_probability:
            watermark=rng.choice(["PREPRINT","SAMPLE","DRAFT",random_text(rng,1,2).upper()])
            wm=Image.new("L",(w,h),0); wm_draw=ImageDraw.Draw(wm)
            wm_font=font(rng.randint(24,52)*s)
            wm_draw.text((rng.randint(0,max(0,w//3)),rng.randint(0,max(0,h-60*s))),
                         watermark,fill=255,font=wm_font)
            if rng.random()<.6:
                wm=wm.rotate(rng.choice([-90,-35,35,90]),Image.Resampling.BICUBIC,expand=False,fillcolor=0)
            watermark_color=(90,90,90) if bg_kind=="dark" else (205,205,205)
            image.paste(watermark_color,mask=wm)
            binary_wm=wm.point(lambda value: 255 if value>8 else 0)
            occluder=ImageChops.lighter(occluder,binary_wm)
            occluder_meta.append({"type":"watermark","text":watermark})
        visible=ImageChops.invert(occluder)
        masks=[ImageChops.multiply(mask,visible) for mask in masks]
        meta={"background":bg_kind,"axis_style":axis_style,"grid_style":grid_style,"title":title,
              "xlabel":xlabel,"ylabel":ylabel,"legend":legend,"occluders":occluder_meta,
              "watermark":watermark,"page_layout":page_layout,
              "multi_panel":False,"panels":[],
              "hard_negative_count":hard_negative_count,
              "mask_semantics":"visible curve pixels only","curves":curve_meta}
        return image,masks,meta

    def generate_one(self,index: int,split: str,out: Path) -> dict:
        seed=self.cfg.seed+index*1_000_003; rng=random.Random(seed); np_rng=np.random.default_rng(seed)
        image,masks,meta=self._render_base(rng,np_rng)
        image,masks,geometry=transform_pair(image,masks,rng,(self.cfg.width,self.cfg.height),
            self.cfg.crop_probability,self.cfg.rotation_probability,self.cfg.perspective_probability,
            self.cfg.crop_min_keep,self.cfg.rotation_max_degrees,
            self.cfg.perspective_max_strength)
        image,degradation=degrade(image,rng,np_rng,self.cfg.degradation_strength) if rng.random()<self.cfg.degradation_probability else (image,{"applied":[]})
        quality=rng.randint(self.cfg.min_jpeg_quality,self.cfg.max_jpeg_quality)
        buffer=io.BytesIO(); image.save(buffer,"JPEG",quality=quality,subsampling=rng.choice([0,1,2]))
        buffer.seek(0); image=Image.open(buffer).convert("RGB"); name=f"{index:08d}"
        _atomic_save_image(image,out/"images"/split/f"{name}.jpg","JPEG",quality=95)
        inst=np.zeros((self.cfg.height,self.cfg.width),dtype=np.uint16)
        sample_mask_dir=out/"curve_masks"/split/name
        sample_mask_dir.mkdir(parents=True,exist_ok=True)
        visible_curves=[]
        for source_curve,m in zip(meta["curves"],masks):
            binary=m.point(lambda v: 255 if v>0 else 0)
            bbox,area=mask_bbox(binary)
            if area==0:
                continue
            instance_id=len(visible_curves)+1
            mask_rel=f"curve_masks/{split}/{name}/curve_{instance_id:03d}.png"
            _atomic_save_image(binary,out/mask_rel,"PNG")
            arr=np.asarray(binary)>0
            # Auxiliary ID map cannot express overlap; independent masks above can.
            inst[arr]=instance_id
            curve={**source_curve,"source_id":source_curve["id"],"id":instance_id,
                   "mask":mask_rel,"bbox":bbox,"area":area}
            visible_curves.append(curve)
        _atomic_save_image(
            Image.fromarray((inst>0).astype(np.uint8)*255,"L"),
            out/"semantic_masks"/split/f"{name}.png",
            "PNG",
        )
        _atomic_save_image(Image.fromarray(inst),out/"instance_masks"/split/f"{name}.png","PNG")
        generated_curve_count=len(meta["curves"])
        meta["curves"]=visible_curves
        result={"id":index,"split":split,"seed":seed,"image":f"images/{split}/{name}.jpg",
                "semantic_mask":f"semantic_masks/{split}/{name}.png","instance_mask":f"instance_masks/{split}/{name}.png",
                "width":self.cfg.width,"height":self.cfg.height,"jpeg_quality":quality,"geometry":geometry,
                "degradation":degradation,"curve_count":len(visible_curves),
                "generated_curve_count":generated_curve_count,**meta}
        _atomic_write_text(
            out/"metadata"/split/f"{name}.json",
            json.dumps(result,ensure_ascii=False,indent=2),
        )
        return result

    def generate(self,out_dir: str|Path,count: int,val_fraction: float=.1,test_fraction: float=.1,
                 workers: int=1,resume: bool=False) -> dict:
        out=Path(out_dir)
        if count<1: raise ValueError("count must be positive")
        if val_fraction<0 or test_fraction<0 or val_fraction+test_fraction>=1:
            raise ValueError("val/test fractions must be >= 0 and sum to less than 1")
        if workers<1: raise ValueError("workers must be positive")
        for folder in ("images","semantic_masks","instance_masks","curve_masks","metadata"):
            for split in ("train","val","test"): (out/folder/split).mkdir(parents=True,exist_ok=True)
        n_test=round(count*test_fraction); n_val=round(count*val_fraction)
        n_train=count-n_val-n_test
        state_path=out/"generation_state.json"
        requested_state={
            "format_version":5,
            "count":count,
            "val_fraction":val_fraction,
            "test_fraction":test_fraction,
            "splits":{"train":n_train,"val":n_val,"test":n_test},
            "config":self.cfg.to_dict(),
        }
        if state_path.exists():
            existing_state=json.loads(state_path.read_text(encoding="utf-8"))
            comparable={key:existing_state.get(key) for key in requested_state}
            if comparable != requested_state:
                raise ValueError(
                    f"Existing generation at {out} has incompatible settings; "
                    "use a new output directory"
                )
            if not resume and existing_state.get("status")!="completed":
                raise ValueError(
                    f"Interrupted generation exists at {out}; rerun with --resume"
                )
            if not resume and existing_state.get("status")=="completed":
                raise ValueError(
                    f"Completed dataset already exists at {out}; use --resume to verify/reuse it"
                )
        requested_state["status"]="running"
        _atomic_write_text(state_path,json.dumps(requested_state,ensure_ascii=False,indent=2))
        manifest=[]
        reused_existing=0
        jobs=[]
        for i in range(count):
            split="train" if i<n_train else "val" if i<n_train+n_val else "test"
            existing=self._load_complete_sample(out,i,split) if resume else None
            if existing is not None:
                manifest.append(existing)
                reused_existing+=1
            else:
                jobs.append((self.cfg.to_dict(),str(out),i,split))
        if workers==1:
            for completed,(_,_,i,split) in enumerate(jobs,1):
                manifest.append(self.generate_one(i,split,out))
                if completed%250==0 or completed==len(jobs):
                    print(f"generated {completed}/{len(jobs)} new samples",flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                generated=pool.map(_generate_job,jobs,chunksize=min(8,max(1,len(jobs)//max(1,workers*32))))
                for completed,result in enumerate(generated,1):
                    manifest.append(result)
                    if completed%250==0 or completed==len(jobs):
                        print(f"generated {completed}/{len(jobs)} new samples",flush=True)
        manifest.sort(key=lambda item:item["id"])
        summary={"format_version":5,"count":count,"splits":{s:sum(x["split"]==s for x in manifest) for s in ("train","val","test")},
                 "target_format":"one independent binary PNG per visible curve",
                 "mask_semantics":"visible except non-curve occluders; curve-curve intersections are multi-label",
                 "classes":{"0":"background","255":"curve in curve_masks"},
                 "generated_now":len(jobs),"reused_existing":reused_existing,
                 "config":self.cfg.to_dict()}
        _atomic_write_text(out/"dataset.json",json.dumps(summary,ensure_ascii=False,indent=2))
        for split in ("train","val","test"):
            rows=[json.dumps(x,ensure_ascii=False) for x in manifest if x["split"]==split]
            payload="\n".join(rows)+(chr(10) if rows else "")
            _atomic_write_text(out/f"{split}.jsonl",payload)
        completed_state={**requested_state,"status":"completed"}
        _atomic_write_text(state_path,json.dumps(completed_state,ensure_ascii=False,indent=2))
        return summary

    @staticmethod
    def _load_complete_sample(out: Path,index: int,split: str) -> dict|None:
        name=f"{index:08d}"
        metadata_path=out/"metadata"/split/f"{name}.json"
        try:
            item=json.loads(metadata_path.read_text(encoding="utf-8"))
            if item.get("id")!=index or item.get("split")!=split:
                return None
            required=(
                out/item["image"],
                out/item["semantic_mask"],
                out/item["instance_mask"],
            )
            if not all(path.is_file() and path.stat().st_size>0 for path in required):
                return None
            curves=item.get("curves",[])
            if item.get("curve_count")!=len(curves):
                return None
            if not all(
                (out/curve["mask"]).is_file() and (out/curve["mask"]).stat().st_size>0
                for curve in curves
            ):
                return None
            return item
        except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError):
            return None


def _generate_job(job: tuple[dict,str,int,str]) -> dict:
    """Top-level worker for Windows spawn-based multiprocessing."""
    cfg,out,index,split=job
    return DatasetGenerator(GeneratorConfig(**cfg)).generate_one(index,split,Path(out))
