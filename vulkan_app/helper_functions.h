#pragma once

#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include <GLFW/glfw3.h>
#ifdef __APPLE__
#pragma clang diagnostic pop
#endif

struct WindowParameters
{
#ifdef VK_USE_PLATFORM_WIN32_KHR
	HINSTANCE HInstance;
	HWND      HWnd;
#elif VK_USE_PLATFORM_XLIB_KHR
	Display*  Dpy;
	Window    Window;
#elif VK_USE_PLATFORM_XCB_KHR
	xcb_connection_t* Connection;
	xcb_window_t*     Window;
#endif
};

struct QueueInfo {
	uint32_t           FamilyIndex;
	std::vector<float> Priorities;
};	

struct PresentInfo {
	vk::SwapchainKHR Swapchain;
	uint32_t         ImageIndex;
};


struct WaitSemaphoreInfo {
	vk::Semaphore          Semaphore;
	vk::PipelineStageFlags WaitingStage;
};

template<typename T>
bool AreAllFlagsSet(T mask, T flags)
{
	return (mask & flags) == mask;
}

template<typename T>
T MinValue(const T& a, const T& b)
{
	return a < b ? a : b;
}

template<typename T>
T MaxValue(const T& a, const T& b)
{
	return a >= b ? a : b;
}

bool InstanceSupportsExtensions(const std::vector<const char*>& desired_extensions);

std::vector<vk::PhysicalDevice>
GetPhysicalDevicesWithExtensions(
	const std::vector<vk::PhysicalDevice>& available_devices,
	const std::vector<const char*>& desired_extensions);

std::vector<vk::PhysicalDevice>
GetPhysicalDevicesWithQueueFamilyProperties(
	const std::vector<vk::PhysicalDevice>& available_devices,
	const vk::QueueFamilyProperties& desired_queue_family_props,
	const vk::SurfaceKHR* presentation_surface);

std::vector<QueueInfo>
GetQueueInfosWithQueueFamilyProperties(
	const vk::PhysicalDevice& physical_device,
	const vk::QueueFamilyProperties& desired_queue_family_props);

