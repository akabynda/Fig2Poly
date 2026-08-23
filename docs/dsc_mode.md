# CurveForge DSC mode

The dedicated DSC profile generates complete differential-scanning-calorimetry
figures rather than selecting a generic peak function for an otherwise random
plot.

```powershell
python -m curveforge --config configs/dsc_v1.json --count 1000 --output dataset_dsc --workers 8
```

For a one-off domain override, use `--plot-domain dsc`. The profile is preferred
because it also selects DSC-appropriate degradation, layout, and annotation
probabilities.

Each `dsc_trace` is composed from:

- a slowly varying linear/quadratic baseline;
- zero or more smooth heat-capacity steps;
- asymmetric generalized-Gaussian thermal events;
- weak correlated measurement noise.

All traces in a figure share one display scale. A figure can use an overlay or
vertically stacked layout, and related series are mutated from a shared latent
thermal-event template. Narrow events receive adaptive sampling around their
centres so peaks only a few pixels wide are retained.

DSC multipanel figures render every panel directly at its final pixel size;
full-size plots are never rendered and then shrunk. Layouts include top/bottom,
left/right, staggered, a large top or bottom panel paired with two smaller
panels, three rows, and grids for four to six panels.

DSC-specific metadata includes `dsc_layout`, `dsc_polarity`,
`dsc_dense_events`, and `dsc_related_series`. Per-curve `function_parameters`
contains the baseline, steps, events, and noise level used to produce the trace.

Important profile controls:

- `dsc_dense_probability`: probability of a rare many-event figure;
- `dsc_max_events`: upper event count for dense traces;
- `related_curves_probability`: probability that series share a latent event template;
- `multi_panel_probability`: probability of a multi-panel DSC figure;
- `min_curves` / `max_curves`: traces per single panel.

Typography has independent output-pixel ranges:

- `dsc_tick_font_min` / `dsc_tick_font_max`: temperature tick labels;
- `dsc_axis_font_min` / `dsc_axis_font_max`: axis labels and titles;
- `dsc_curve_label_font_min` / `dsc_curve_label_font_max`: labels attached directly to traces;
- `dsc_annotation_font_min` / `dsc_annotation_font_max`: peak-temperature annotations;
- `dsc_direct_labels_probability`: probability of labels placed at random left, middle, or right points along each curve when a legend is not used.

The upper size is automatically capped for a physically small panel so a large
configured font cannot consume the whole plot.

Document-layout augmentation is controlled independently:

- `dsc_page_layout_probability`: embed the DSC figure into a larger article page;
- `dsc_multipanel_page_probability`: independently embed a multipanel figure into an article page;
- `dsc_plot_min_fraction`: minimum outer figure size relative to the PNG;
- `dsc_surrounding_text_probability`: add headers and paragraph-like text bands;
- `dsc_caption_probability`: add an article-style figure caption;
- `dsc_foreign_graphics_probability`: add a non-target table or mini-plot;
- `dsc_watermark_probability`: overlay a watermark and remove its pixels from target masks.
- `hard_negatives_probability`: add reference lines, integration baselines, or onset tangents;
- `occlusion_probability`: add small opaque text boxes over the plot.

Surrounding text, tables, foreign plots, captions, and watermarks are hard
negatives. Only DSC traces are written to `curve_masks`; every added line, label,
and opaque text box is also removed from the visible-curve masks.

Multipanel and document layout are sampled independently. Consequently the
dataset can contain clean single figures, documented single figures, clean
multipanel figures, and documented multipanel figures. Text, captions, foreign
graphics, and watermarks are independent optional layers within either document
variant.

Nominal rendered curve widths are 1, 2, or 3 output pixels, with 1 pixel being
the most common value.
