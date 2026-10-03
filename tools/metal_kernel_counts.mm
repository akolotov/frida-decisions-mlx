// Host-side profiling only. This does not implement or modify GPU kernels.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
std::atomic<bool> recording{false};
std::mutex lock;
std::unordered_map<void*, std::string> pipelines;
std::unordered_map<std::string, size_t> counts;
thread_local std::string current;
IMP originalPipeline, originalThreads, originalGroups, originalFunction, originalDescriptor;

void remember(id state, id<MTLFunction> function) {
    if (!state || !function) return;
    std::lock_guard<std::mutex> guard(lock);
    pipelines[(__bridge void*)state] = function.name.UTF8String;
}

id makeFunction(id device, SEL selector, id<MTLFunction> function, NSError** error) {
    id state = ((id (*)(id, SEL, id, NSError**))originalFunction)(device, selector, function, error);
    remember(state, function);
    return state;
}

id makeDescriptor(id device, SEL selector, MTLComputePipelineDescriptor* descriptor,
                  MTLPipelineOption options, MTLComputePipelineReflection** reflection, NSError** error) {
    id state = ((id (*)(id, SEL, id, MTLPipelineOption, id*, NSError**))originalDescriptor)(
        device, selector, descriptor, options, (id*)reflection, error);
    remember(state, descriptor.computeFunction);
    return state;
}

void setPipeline(id encoder, SEL selector, id<MTLComputePipelineState> pipeline) {
    if (recording) {
        std::lock_guard<std::mutex> guard(lock);
        auto it = pipelines.find((__bridge void*)pipeline);
        current = it == pipelines.end() ? (pipeline.label ? pipeline.label.UTF8String : "unknown") : it->second;
    }
    ((void (*)(id, SEL, id))originalPipeline)(encoder, selector, pipeline);
}

void dispatchThreads(id encoder, SEL selector, MTLSize grid, MTLSize group) {
    if (recording) { std::lock_guard<std::mutex> guard(lock); ++counts[current]; }
    ((void (*)(id, SEL, MTLSize, MTLSize))originalThreads)(encoder, selector, grid, group);
}

void dispatchGroups(id encoder, SEL selector, MTLSize grid, MTLSize group) {
    if (recording) { std::lock_guard<std::mutex> guard(lock); ++counts[current]; }
    ((void (*)(id, SEL, MTLSize, MTLSize))originalGroups)(encoder, selector, grid, group);
}

IMP replace(Class cls, SEL selector, IMP replacement) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) return nullptr;
    // Add to the concrete class before replacing an inherited method.
    IMP old = method_getImplementation(method);
    if (!class_addMethod(cls, selector, replacement, method_getTypeEncoding(method)))
        method_setImplementation(class_getInstanceMethod(cls, selector), replacement);
    return old;
}
}

extern "C" void frida_profile_install() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        Class dc = object_getClass(device), ec = object_getClass(encoder);
        originalFunction = replace(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)makeFunction);
        originalDescriptor = replace(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)makeDescriptor);
        originalPipeline = replace(ec, @selector(setComputePipelineState:), (IMP)setPipeline);
        originalThreads = replace(ec, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)dispatchThreads);
        originalGroups = replace(ec, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)dispatchGroups);
        [encoder endEncoding];
    }
}

extern "C" void frida_profile_start() {
    std::lock_guard<std::mutex> guard(lock);
    counts.clear();
    recording = true;
}

extern "C" void frida_profile_stop(const char* output) {
    recording = false;
    @autoreleasepool {
        std::lock_guard<std::mutex> guard(lock);
        NSMutableDictionary* dictionary = [NSMutableDictionary dictionary];
        for (const auto& entry : counts)
            dictionary[@(entry.first.c_str())] = @(entry.second);
        NSData* data = [NSJSONSerialization dataWithJSONObject:dictionary options:NSJSONWritingPrettyPrinted error:nil];
        [data writeToFile:@(output) atomically:YES];
    }
}
