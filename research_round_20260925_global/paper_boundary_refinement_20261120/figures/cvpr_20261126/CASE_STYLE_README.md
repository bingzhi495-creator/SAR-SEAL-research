# Restyling the qualitative case figures

The case artwork uses a common sans-serif hierarchy, pale rules for the control columns, blue emphasis for the evaluated method, and signed teal and orange cues. The matched columns and sign-specific selections retain their original meaning. The public SVG retains the original shape coordinates, text and method order. Its single-page PDF is printed from the editable SVG using Chromium so dashed reference marks and text remain visible as vector artwork.

Run the script with Python 3.10 or later, PyMuPDF, lxml, Matplotlib and a Chromium-based browser. Pass a fresh output directory.

~~~sh
python style_case_panels.py --public figures --out case_style
~~~

The default browser is Microsoft Edge in its standard Windows installation directory. Pass `--edge` with the path to another compatible executable if needed. A separate optional `--private` source directory admits the completed native-pixel figure PDFs for local use. At rendering, the script verifies the native raster digest and placement for each crop. The checked native crops, original annotation traces and study result figures remain in their original locations. Only vector rules are added in their margins.

The five public output PDFs contain no raster images. The five local output PDFs preserve all 54 original image placements and all labels. The accompanying validation report records the checks for decimal scores and native placements. Only the explicitly identified public SVG, PDF and source files belong in the shared release.
