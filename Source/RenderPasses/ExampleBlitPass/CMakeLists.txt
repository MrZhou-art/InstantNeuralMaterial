add_plugin(ExampleBlitPass)

target_sources(ExampleBlitPass PRIVATE
    ExampleBlitPass.cpp
    ExampleBlitPass.h
)

target_source_group(ExampleBlitPass "RenderPasses")
