add_plugin(ExampleBlitPass)

target_sources(ExampleBlitPass PRIVATE
    ExampleBlitPass.cpp
    ExampleBlitPass.h
    wireframe_shader.slang
)
target_copy_shaders(ExampleBlitPass RenderPasses/ExampleBlitPass)

target_source_group(ExampleBlitPass "RenderPasses")
