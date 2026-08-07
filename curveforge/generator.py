from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import io
import json
import os
import random

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
        }

    def _render_base(self,rng: random.Random,np_rng: np.random.Generator,
                     allow_multipanel: bool=True,force_plot_full: bool=False,
                     forced_bg_kind: str|None=None,
                     forced_curve_max: int|None=None):
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
