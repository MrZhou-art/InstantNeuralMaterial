from falcor import *

def render_graph_WireframePass():
    g = RenderGraph("ExampleBlitPass")
    WireframePass = createPass("ExampleBlitPass")
    g.addPass(WireframePass, "ExampleBlitPass")
    g.markOutput("ExampleBlitPass.output")
    return g

WireframPass = render_graph_WireframePass()
try: m.addGraph(WireframPass)
except NameError: None