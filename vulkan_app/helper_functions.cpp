
#include "helper_functions.h"

bool InstanceSupportsExtensions(const std::vector<const char*>& desired_extensions)
{
	uint32_t glfw_required_instance_extension_count = 0;
	const char** glfw_required_instance_extensions =
		glfwGetRequiredInstanceExtensions(&glfw_required_instance_extension_count);
	
	uint32_t glfw_required_instance_extensions_found_count = 0;

	std::vector<vk::ExtensionProperties> available_instance_extensions
		= vk::enumerateInstanceExtensionProperties();

	for (auto& desired_extension : desired_extensions)
	{
		bool desired_extension_found = false;

		for (uint32_t i = 0; i < glfw_required_instance_extension_count; ++i)
		{
			if (strcmp(desired_extension, glfw_required_instance_extensions[i]) == 0)
			{
				++glfw_required_instance_extensions_found_count;
				break;
			}
		}

		for (auto& available_extension : available_instance_extensions)
		{
			if (strcmp(desired_extension, available_extension.extensionName) == 0)
			{
				desired_extension_found = true;
				break;
			}
		}

		if (!desired_extension_found)
		{
			return false;
		}
	}

	if (glfw_required_instance_extensions_found_count < glfw_required_instance_extension_count)
	{
		return false;
	}

	return true;
}

std::vector<vk::PhysicalDevice>
GetPhysicalDevicesWithExtensions(
	const std::vector<vk::PhysicalDevice>& available_devices,
	const std::vector<const char*>& desired_extensions)
{
	std::vector<vk::PhysicalDevice> devices;

	for (auto& physical_device : available_devices)
	{
		bool has_desired_extensions = true;

		const auto available_device_extensions = physical_device.enumerateDeviceExtensionProperties();

		for (auto& desired_extension : desired_extensions)
		{
			bool has_required_extension = false;

			for (auto& available_extension : available_device_extensions)
			{
				if (strcmp(desired_extension, available_extension.extensionName) == 0)
				{
					has_required_extension = true;
					break;
				}
			}

			if (!has_required_extension)
			{
				has_desired_extensions = false;
				break;
			}
		}

		if (has_desired_extensions)
		{
			devices.push_back(physical_device);
		}
	}

	return devices;
}

std::vector<vk::PhysicalDevice>
GetPhysicalDevicesWithQueueFamilyProperties(
	const std::vector<vk::PhysicalDevice>& available_devices,
	const vk::QueueFamilyProperties& desired_queue_family_props,
	const vk::SurfaceKHR* presentation_surface)
{
	std::vector<vk::PhysicalDevice> devices;

	for (auto& physical_device : available_devices)
	{
		bool has_desired_queue_family_props = false;

		const auto queue_family_props = physical_device.getQueueFamilyProperties();

		uint32_t queue_family_index = 0;
		for (auto& queue_family_prop : queue_family_props)
		{
			if (queue_family_prop.queueCount > 0 &&
				AreAllFlagsSet(desired_queue_family_props.queueFlags, queue_family_prop.queueFlags))
			{
				vk::Bool32 presentation_supported = VK_FALSE;

				if (presentation_surface)
				{
					presentation_supported =
						physical_device.getSurfaceSupportKHR(queue_family_index, *presentation_surface);
				}

				if (presentation_surface == nullptr || presentation_supported)
				{
					has_desired_queue_family_props = true;
					break;
				}
			}

			++queue_family_index;
		}

		if (has_desired_queue_family_props)
		{
			devices.push_back(physical_device);
		}
	}

	return devices;
}

std::vector<QueueInfo>
GetQueueInfosWithQueueFamilyProperties(
	const vk::PhysicalDevice& physical_device,
	const vk::QueueFamilyProperties& desired_queue_family_props)
{
	std::vector<QueueInfo> queue_infos;

	const auto queue_family_props = physical_device.getQueueFamilyProperties();

	int family_index = 0;
	for (auto& queue_family_prop : queue_family_props)
	{
		if (queue_family_prop.queueCount > 0 &&
			queue_family_prop.queueFlags & desired_queue_family_props.queueFlags)
		{
			QueueInfo queue_info;
			queue_info.FamilyIndex = family_index;
			queue_info.Priorities.resize(queue_family_prop.queueCount);
			for (auto& priority : queue_info.Priorities)
			{
				priority = 1.0f;
			}

			queue_infos.push_back(queue_info);
		}

		++family_index;
	}

	return queue_infos;
}

