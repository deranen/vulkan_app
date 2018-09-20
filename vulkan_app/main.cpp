
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <unordered_map>

#define GLM_FORCE_RADIANS
#define GML_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/hash.hpp"

#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#ifdef __APPLE__
#pragma clang diagnostic pop
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "helper_functions.h"
#include "file_io.h"

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texcoord;

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color && texcoord == other.texcoord;
	}

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription binding_desc = {};
		binding_desc.binding = 0;
		binding_desc.stride = sizeof(Vertex);
		binding_desc.inputRate = vk::VertexInputRate::eVertex;

		return binding_desc;
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<vk::VertexInputAttributeDescription, 3> attribute_desc = {};
		attribute_desc[0].binding = 0;
		attribute_desc[0].location = 0;
		attribute_desc[0].format = vk::Format::eR32G32B32Sfloat;
		attribute_desc[0].offset = offsetof(Vertex, pos);

		attribute_desc[1].binding = 0;
		attribute_desc[1].location = 1;
		attribute_desc[1].format = vk::Format::eR32G32B32Sfloat;
		attribute_desc[1].offset = offsetof(Vertex, color);

		attribute_desc[2].binding = 0;
		attribute_desc[2].location = 2;
		attribute_desc[2].format = vk::Format::eR32G32Sfloat;
		attribute_desc[2].offset = offsetof(Vertex, texcoord);

		return attribute_desc;
	}
};


namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texcoord) << 1);
		}
	};
}

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

#ifdef __APPLE__
		double scale_factor = 0.5;
#else
		double scale_factor = 1.0;
#endif
		
		m_window = glfwCreateWindow((int)(scale_factor*WindowWidth),
									(int)(scale_factor*WindowHeight),
									"Vulkan", nullptr, nullptr);

		glfwSetWindowUserPointer(m_window, this);
		glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->m_framebuffer_resized = true;
	}

	void createInstance()
	{
		vk::ApplicationInfo app_info = {};
		app_info.pApplicationName = "Vulkan Application";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "Vulkan Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		uint32_t glfw_extension_count = 0;
		const char** glfw_extensions = nullptr;

		glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

		vk::InstanceCreateInfo instance_ci = {};
		instance_ci.flags = {};
		instance_ci.pApplicationInfo = &app_info;
		instance_ci.enabledLayerCount = 0;
		instance_ci.ppEnabledLayerNames = nullptr;
		instance_ci.enabledExtensionCount = glfw_extension_count;
		instance_ci.ppEnabledExtensionNames = glfw_extensions;

		m_instance = vk::createInstance(instance_ci);
	}

	void createSurface()
	{
		assert(m_instance && m_window);

		if (glfwCreateWindowSurface(m_instance, m_window, nullptr, (VkSurfaceKHR*)&m_surface) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create window surface.");
		}
	}

	void pickPhysicalDevice()
	{
		assert(m_instance && m_surface);

		const auto available_physical_devices = m_instance.enumeratePhysicalDevices();

		std::vector<const char*> desired_physical_device_extensions;
		desired_physical_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		auto desired_devices = GetPhysicalDevicesWithExtensions(
			available_physical_devices, desired_physical_device_extensions);

		if (desired_devices.empty())
		{
			throw std::runtime_error("Could not find physical devices with desired device extensions.");
		}

		vk::QueueFamilyProperties desired_queue_family_props = {};
		desired_queue_family_props.queueFlags |= vk::QueueFlagBits::eGraphics;

		desired_devices = GetPhysicalDevicesWithQueueFamilyProperties(
			desired_devices, desired_queue_family_props, &m_surface);

		if (desired_devices.empty())
		{
			throw std::runtime_error("Could not find physical devices with desired queue family properties.");
		}

//		for (const auto& device : desired_devices)
//		{
//			auto features = device.getFeatures();
//			if (features.tessellationShader && features.geometryShader)
//			{
//				m_physical_device = device;
//				break;
//			}
//		}
		
		m_physical_device = desired_devices[0];

		if (!m_physical_device)
		{
			throw std::runtime_error("Could not find physical devices with desired features.");
		}

		m_msaa_samples = getMaxUsableSampleCount();
	}

	void createLogicalDevice()
	{
		assert(m_physical_device);

		vk::PhysicalDeviceFeatures device_features = {};
		device_features.samplerAnisotropy = VK_TRUE;

		vk::QueueFamilyProperties desired_queue_family_props = {};
		desired_queue_family_props.queueFlags |= vk::QueueFlagBits::eGraphics;

		m_queue_infos = GetQueueInfosWithQueueFamilyProperties(m_physical_device, desired_queue_family_props);

		assert(!m_queue_infos.empty());

		std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;

		for (const auto& queue_info : m_queue_infos)
		{
			vk::DeviceQueueCreateInfo device_queue_create_info = {};
			device_queue_create_info.flags = {};
			device_queue_create_info.queueFamilyIndex = queue_info.FamilyIndex;
			device_queue_create_info.queueCount = (uint32_t)queue_info.Priorities.size();
			device_queue_create_info.pQueuePriorities =
				queue_info.Priorities.size() > 0 ? queue_info.Priorities.data() : nullptr;

			device_queue_create_infos.push_back(device_queue_create_info);
		}

		std::vector<char const*> desired_device_extensions;
		desired_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		vk::DeviceCreateInfo device_create_info = {};
		device_create_info.flags = {};
		device_create_info.queueCreateInfoCount = (uint32_t)device_queue_create_infos.size();
		device_create_info.pQueueCreateInfos =
			device_queue_create_infos.size() > 0 ? device_queue_create_infos.data() : nullptr;
		device_create_info.enabledLayerCount = 0;
		device_create_info.ppEnabledLayerNames = nullptr;
		device_create_info.enabledExtensionCount = (uint32_t)desired_device_extensions.size();
		device_create_info.ppEnabledExtensionNames =
			desired_device_extensions.size() > 0 ? desired_device_extensions.data() : nullptr;
		device_create_info.pEnabledFeatures = &device_features;

		m_device = m_physical_device.createDevice(device_create_info);

		if (m_queue_infos.size() == 1)
		{
			m_graphics_queue = m_device.getQueue(m_queue_infos[0].FamilyIndex, 0);
			m_present_queue  = m_device.getQueue(m_queue_infos[0].FamilyIndex, 0);
		}
		else if (m_queue_infos.size() >= 2)
		{
			m_graphics_queue = m_device.getQueue(m_queue_infos[0].FamilyIndex, 0);
			m_present_queue  = m_device.getQueue(m_queue_infos[1].FamilyIndex, 0);
		}
	}

	void createSwapchain()
	{
		assert(m_physical_device && m_device && m_surface);

		std::vector<vk::PresentModeKHR> available_present_modes =
			m_physical_device.getSurfacePresentModesKHR(m_surface);

		if (available_present_modes.empty())
		{
			throw std::runtime_error("Could not find any surface present modes on physical device.");
		}

		vk::SurfaceCapabilitiesKHR surface_capabilities = m_physical_device.getSurfaceCapabilitiesKHR(m_surface);

		uint32_t swapchain_image_count = surface_capabilities.minImageCount + 1;
		if (surface_capabilities.maxImageCount > 0 &&
			swapchain_image_count > surface_capabilities.maxImageCount)
		{
			swapchain_image_count = surface_capabilities.maxImageCount;
		}

		vk::ImageUsageFlags image_usage = vk::ImageUsageFlagBits::eColorAttachment;
		vk::ImageUsageFlags desired_image_usage = vk::ImageUsageFlagBits::eColorAttachment;
		if (AreAllFlagsSet(desired_image_usage, surface_capabilities.supportedUsageFlags))
		{
			image_usage = desired_image_usage;
		}

		vk::Extent2D swapchain_image_size = {};
		if (surface_capabilities.currentExtent.width == 0xFFFFFFFF)
		{
			swapchain_image_size = vk::Extent2D{ 640, 480 };

			swapchain_image_size.width = MaxValue(swapchain_image_size.width, surface_capabilities.minImageExtent.width);
			swapchain_image_size.width = MinValue(swapchain_image_size.width, surface_capabilities.maxImageExtent.width);
			swapchain_image_size.height = MaxValue(swapchain_image_size.height, surface_capabilities.minImageExtent.height);
			swapchain_image_size.height = MinValue(swapchain_image_size.height, surface_capabilities.maxImageExtent.height);
		}
		else
		{
			swapchain_image_size = surface_capabilities.currentExtent;
		}

		std::vector<vk::SurfaceFormatKHR> surface_formats = m_physical_device.getSurfaceFormatsKHR(m_surface);

		vk::SurfaceFormatKHR surface_format = {};
		vk::SurfaceFormatKHR desired_surface_format = {};
#ifdef __APPLE__
		desired_surface_format.format = vk::Format::eB8G8R8A8Unorm;
#elif _WIN32
		desired_surface_format.format = vk::Format::eR8G8B8A8Unorm;
#endif
		desired_surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;

		if (surface_formats.size() == 1 && surface_formats[0].format == vk::Format::eUndefined)
		{
			surface_format = desired_surface_format;
		}
		else
		{
			for (auto& sf : surface_formats)
			{
				if (desired_surface_format == sf)
				{
					surface_format = desired_surface_format;
					break;
				}
			}

			if (surface_format != desired_surface_format)
			{
				throw std::runtime_error("Desired surface format not supported.");
			}
		}

		vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
		vk::PresentModeKHR desired_present_mode = vk::PresentModeKHR::eMailbox;

		for (auto available_present_mode : available_present_modes)
		{
			if (desired_present_mode == available_present_mode)
			{
				present_mode = desired_present_mode;
				break;
			}
		}

		vk::SwapchainCreateInfoKHR swapchain_create_info = {};
		swapchain_create_info.flags = {};
		swapchain_create_info.surface = m_surface;
		swapchain_create_info.minImageCount = swapchain_image_count;
		swapchain_create_info.imageFormat = surface_format.format;
		swapchain_create_info.imageColorSpace = surface_format.colorSpace;
		swapchain_create_info.imageExtent = swapchain_image_size;
		swapchain_create_info.imageArrayLayers = 1;
		swapchain_create_info.imageUsage = image_usage;

		assert(m_queue_infos.size() > 0);

		if (m_queue_infos.size() == 1)
		{
			std::vector<uint32_t> queue_indices = { m_queue_infos[0].FamilyIndex };

			swapchain_create_info.imageSharingMode = vk::SharingMode::eExclusive;
			swapchain_create_info.queueFamilyIndexCount = 1;
			swapchain_create_info.pQueueFamilyIndices = queue_indices.data();
		}
		else if (m_queue_infos.size() >= 2)
		{
			std::vector<uint32_t> queue_indices = { m_queue_infos[0].FamilyIndex, m_queue_infos[1].FamilyIndex };

			swapchain_create_info.imageSharingMode = vk::SharingMode::eConcurrent;
			swapchain_create_info.queueFamilyIndexCount = (uint32_t)queue_indices.size();
			swapchain_create_info.pQueueFamilyIndices = queue_indices.data();
		}

		swapchain_create_info.preTransform = surface_capabilities.currentTransform;
		swapchain_create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		swapchain_create_info.presentMode = present_mode;
		swapchain_create_info.clipped = (vk::Bool32)VK_TRUE;
		swapchain_create_info.oldSwapchain = nullptr;

		m_swapchain = m_device.createSwapchainKHR(swapchain_create_info);

		m_swapchain_images = m_device.getSwapchainImagesKHR(m_swapchain);

		m_swapchain_image_format = surface_format.format;
		m_swapchain_extent = swapchain_image_size;
	}

	void createImageViews()
	{
		m_swapchain_image_views.resize(m_swapchain_images.size());

		for (size_t i = 0; i < m_swapchain_images.size(); ++i)
		{
			m_swapchain_image_views[i] =
				createImageView(m_swapchain_images[i], m_swapchain_image_format,
					vk::ImageAspectFlagBits::eColor, 1);
		}
	}

	void createRenderPass()
	{
		assert(m_device);

		vk::AttachmentDescription color_attachment = {};
		color_attachment.format = m_swapchain_image_format;
		color_attachment.samples = m_msaa_samples;
		color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
		color_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		color_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		color_attachment.initialLayout = vk::ImageLayout::eUndefined;
		color_attachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentDescription depth_attachment = {};
		depth_attachment.format = findDepthFormat();
		depth_attachment.samples = m_msaa_samples;
		depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		depth_attachment.storeOp = vk::AttachmentStoreOp::eDontCare;
		depth_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		depth_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		depth_attachment.initialLayout = vk::ImageLayout::eUndefined;
		depth_attachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference depth_attachment_ref = {};
		depth_attachment_ref.attachment = 1;
		depth_attachment_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentDescription color_attachment_resolve = {};
		color_attachment_resolve.format = m_swapchain_image_format;
		color_attachment_resolve.samples = vk::SampleCountFlagBits::e1;
		color_attachment_resolve.loadOp = vk::AttachmentLoadOp::eDontCare;
		color_attachment_resolve.storeOp = vk::AttachmentStoreOp::eStore;
		color_attachment_resolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		color_attachment_resolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		color_attachment_resolve.initialLayout = vk::ImageLayout::eUndefined;
		color_attachment_resolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

		vk::AttachmentReference color_attachment_resolve_ref = {};
		color_attachment_resolve_ref.attachment = 2;
		color_attachment_resolve_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass = {};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;
		subpass.pResolveAttachments = &color_attachment_resolve_ref;

		vk::SubpassDependency subpass_dependency = {};
		subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		subpass_dependency.dstSubpass = 0;
		subpass_dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		subpass_dependency.srcAccessMask = (vk::AccessFlags)0;
		subpass_dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		subpass_dependency.dstAccessMask =
			vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

		std::array<vk::AttachmentDescription, 3> attachments =
		{ color_attachment, depth_attachment, color_attachment_resolve };

		vk::RenderPassCreateInfo render_pass_info = {};
		render_pass_info.attachmentCount = (uint32_t)attachments.size();
		render_pass_info.pAttachments = attachments.data();
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &subpass_dependency;

		m_render_pass = m_device.createRenderPass(render_pass_info);
	}

	void createGraphicsPipeline()
	{
		assert(m_device);

		auto vert_shader_code = ReadFile("shaders/vert.spv");
		auto frag_shader_code = ReadFile("shaders/frag.spv");

		vk::ShaderModuleCreateInfo vert_shader_module_ci = {};
		vert_shader_module_ci.codeSize = (uint32_t)vert_shader_code.size();
		vert_shader_module_ci.pCode = reinterpret_cast<const uint32_t*>(vert_shader_code.data());

		vk::ShaderModuleCreateInfo frag_shader_module_ci = {};
		frag_shader_module_ci.codeSize = (uint32_t)frag_shader_code.size();
		frag_shader_module_ci.pCode = reinterpret_cast<const uint32_t*>(frag_shader_code.data());

		vk::ShaderModule vert_shader_module = m_device.createShaderModule(vert_shader_module_ci);
		vk::ShaderModule frag_shader_module = m_device.createShaderModule(frag_shader_module_ci);

		vk::PipelineShaderStageCreateInfo vert_stage_ci = {};
		vert_stage_ci.stage = vk::ShaderStageFlagBits::eVertex;
		vert_stage_ci.module = vert_shader_module;
		vert_stage_ci.pName = "main";

		vk::PipelineShaderStageCreateInfo frag_stage_ci = {};
		frag_stage_ci.stage = vk::ShaderStageFlagBits::eFragment;
		frag_stage_ci.module = frag_shader_module;
		frag_stage_ci.pName = "main";

		vk::PipelineShaderStageCreateInfo shader_stages[] = { vert_stage_ci , frag_stage_ci };

		auto vertex_desc = Vertex::getBindingDescription();
		auto attribute_descs = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertex_info = {};
		vertex_info.vertexBindingDescriptionCount = 1;
		vertex_info.pVertexBindingDescriptions = &vertex_desc;
		vertex_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descs.size());
		vertex_info.pVertexAttributeDescriptions = attribute_descs.data();

		vk::PipelineInputAssemblyStateCreateInfo assembly_info = {};
		assembly_info.topology = vk::PrimitiveTopology::eTriangleList;
		assembly_info.primitiveRestartEnable = VK_FALSE;

		vk::Viewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)m_swapchain_extent.width;
		viewport.height = (float)m_swapchain_extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		vk::Rect2D scissor = {};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = m_swapchain_extent;

		vk::PipelineViewportStateCreateInfo viewport_state = {};
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		vk::PipelineRasterizationStateCreateInfo rasterization = {};
		rasterization.depthClampEnable = VK_FALSE;
		rasterization.rasterizerDiscardEnable = VK_FALSE;
		rasterization.polygonMode = vk::PolygonMode::eFill;
		rasterization.lineWidth = 1.0f;
		rasterization.cullMode = vk::CullModeFlagBits::eBack;
		rasterization.frontFace = vk::FrontFace::eCounterClockwise;
		rasterization.depthBiasEnable = VK_FALSE;
		rasterization.depthBiasConstantFactor = 0.0f;
		rasterization.depthBiasClamp = 0.0f;
		rasterization.depthBiasSlopeFactor = 0.0f;

		vk::PipelineMultisampleStateCreateInfo multisample = {};
		multisample.sampleShadingEnable = VK_FALSE;
		multisample.rasterizationSamples = m_msaa_samples;
		multisample.minSampleShading = 1.0f; // Optional
		multisample.pSampleMask = nullptr; // Optional
		multisample.alphaToCoverageEnable = VK_FALSE; // Optional
		multisample.alphaToOneEnable = VK_FALSE; // Optional

		vk::PipelineColorBlendAttachmentState color_blend_attachment = {};
		color_blend_attachment.colorWriteMask =
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		color_blend_attachment.blendEnable = VK_FALSE;
		color_blend_attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		color_blend_attachment.dstColorBlendFactor = vk::BlendFactor::eZero;
		color_blend_attachment.colorBlendOp = vk::BlendOp::eAdd; // Optional
		color_blend_attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
		color_blend_attachment.dstAlphaBlendFactor = vk::BlendFactor::eZero; // Optional
		color_blend_attachment.alphaBlendOp = vk::BlendOp::eAdd;; // Optional

		vk::PipelineColorBlendStateCreateInfo color_blend = {};
		color_blend.logicOpEnable = VK_FALSE;
		color_blend.logicOp = vk::LogicOp::eCopy; // Optional
		color_blend.attachmentCount = 1;
		color_blend.pAttachments = &color_blend_attachment;
		color_blend.blendConstants[0] = 0.0f; // Optional
		color_blend.blendConstants[1] = 0.0f; // Optional
		color_blend.blendConstants[2] = 0.0f; // Optional
		color_blend.blendConstants[3] = 0.0f; // Optional

		vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
		pipeline_layout_info.setLayoutCount = 1;
		pipeline_layout_info.pSetLayouts = &m_descriptor_set_layout;
		pipeline_layout_info.pushConstantRangeCount = 0; // Optional
		pipeline_layout_info.pPushConstantRanges = nullptr; // Optional

		m_pipeline_layout = m_device.createPipelineLayout(pipeline_layout_info);

		vk::PipelineDepthStencilStateCreateInfo depth_stencil_info;
		depth_stencil_info.depthTestEnable = VK_TRUE;
		depth_stencil_info.depthWriteEnable = VK_TRUE;
		depth_stencil_info.depthCompareOp = vk::CompareOp::eLess;
		depth_stencil_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_info.minDepthBounds = 0.0f; // Optional
		depth_stencil_info.maxDepthBounds = 1.0f; // Optional
		depth_stencil_info.stencilTestEnable = VK_FALSE;
		depth_stencil_info.front = vk::StencilOpState{}; // Optional
		depth_stencil_info.back = vk::StencilOpState{}; // Optional

		vk::GraphicsPipelineCreateInfo pipeline_info = {};
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_info;
		pipeline_info.pInputAssemblyState = &assembly_info;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterization;
		pipeline_info.pMultisampleState = &multisample;
		pipeline_info.pDepthStencilState = &depth_stencil_info;
		pipeline_info.pColorBlendState = &color_blend;
		pipeline_info.pDynamicState = nullptr;
		pipeline_info.layout = m_pipeline_layout;
		pipeline_info.renderPass = m_render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = nullptr;
		pipeline_info.basePipelineIndex = -1;

		m_pipeline = m_device.createGraphicsPipeline(nullptr, pipeline_info);

		m_device.destroyShaderModule(vert_shader_module);
		m_device.destroyShaderModule(frag_shader_module);
	}

	void createFramebuffers()
	{
		m_swapchain_frame_buffers.resize(m_swapchain_image_views.size());

		for (size_t i = 0; i < m_swapchain_image_views.size(); ++i)
		{
			std::array<vk::ImageView, 3> attachments =
			{
				m_color_image_view,
				m_depth_image_view,
				m_swapchain_image_views[i]
			};
				
			vk::FramebufferCreateInfo framebuffer_info = {};
			framebuffer_info.renderPass = m_render_pass;
			framebuffer_info.attachmentCount = (uint32_t)attachments.size();
			framebuffer_info.pAttachments = attachments.data();
			framebuffer_info.width = m_swapchain_extent.width;
			framebuffer_info.height = m_swapchain_extent.height;
			framebuffer_info.layers = 1;

			m_swapchain_frame_buffers[i] = m_device.createFramebuffer(framebuffer_info);
		}
	}

	void createCommandPool()
	{
		assert(m_queue_infos.size() > 0);

		vk::CommandPoolCreateInfo pool_info = {};
		pool_info.queueFamilyIndex = m_queue_infos[0].FamilyIndex;
		pool_info.flags = (vk::CommandPoolCreateFlags)0;

		m_command_pool = m_device.createCommandPool(pool_info);
	}

	void createCommandBuffers()
	{
		vk::CommandBufferAllocateInfo alloc_info = {};
		alloc_info.commandPool = m_command_pool;
		alloc_info.level = vk::CommandBufferLevel::ePrimary;
		alloc_info.commandBufferCount = (uint32_t)m_swapchain_frame_buffers.size();

		m_command_buffers = m_device.allocateCommandBuffers(alloc_info);

		for (size_t i = 0; i < m_command_buffers.size(); ++i)
		{
			vk::CommandBufferBeginInfo begin_info = {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
			begin_info.pInheritanceInfo = nullptr;

			m_command_buffers[i].begin(begin_info);

			vk::RenderPassBeginInfo render_pass_info = {};
			render_pass_info.renderPass = m_render_pass;
			render_pass_info.framebuffer = m_swapchain_frame_buffers[i];
			render_pass_info.renderArea.offset = vk::Offset2D{ 0, 0 };
			render_pass_info.renderArea.extent = m_swapchain_extent;

			std::array<vk::ClearValue, 2> clear_values = {};
			clear_values[0].color = vk::ClearColorValue(std::array<float, 4>({ 0.0f, 0.0f, 0.0f, 1.0f }));
			clear_values[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

			render_pass_info.clearValueCount = (uint32_t)clear_values.size();
			render_pass_info.pClearValues = clear_values.data();

			m_command_buffers[i].beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
			m_command_buffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

			vk::Buffer vertex_buffers[] = { m_vertex_buffer };
			vk::DeviceSize offsets[] = { 0 };
			m_command_buffers[i].bindVertexBuffers(0, 1, vertex_buffers, offsets);
			m_command_buffers[i].bindIndexBuffer(m_index_buffer, 0, vk::IndexType::eUint32);
			m_command_buffers[i].bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1, &m_descriptor_sets[i], 0, nullptr);

			m_command_buffers[i].drawIndexed(static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);
			m_command_buffers[i].endRenderPass();

			m_command_buffers[i].end();
		}
	}

	void createSemaphoresAndFences()
	{
		m_image_available_semaphores.resize(MaxFramesInFlight);
		m_render_finished_semaphores.resize(MaxFramesInFlight);
		m_in_flight_fences.resize(MaxFramesInFlight);

		vk::SemaphoreCreateInfo semaphore_info = {};
		vk::FenceCreateInfo fence_info = {};
		fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

		for (int i = 0; i < MaxFramesInFlight; ++i)
		{
			m_image_available_semaphores[i] = m_device.createSemaphore(semaphore_info);
			m_render_finished_semaphores[i] = m_device.createSemaphore(semaphore_info);
			m_in_flight_fences[i] = m_device.createFence(fence_info);
		}
	}

	void recreateSwapchain()
	{
		assert(m_device && m_window);

		int width = 0, height = 0;
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(m_window, &width, &height);
			glfwWaitEvents();
		}

		m_device.waitIdle();

		cleanupSwapchain();

		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createColorResources();
		createDepthResources();
		createFramebuffers();
		createCommandBuffers();
	}

	void createVertexBuffer()
	{
		assert(m_device && m_vertices.size() > 0);

		vk::DeviceSize buffer_size = sizeof(m_vertices[0])*m_vertices.size();

		vk::Buffer staging_buffer = nullptr;
		vk::DeviceMemory staging_buffer_memory = nullptr;

		createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			staging_buffer, staging_buffer_memory);

		void* data = m_device.mapMemory(staging_buffer_memory, 0, buffer_size, (vk::MemoryMapFlags)0);

		memcpy(data, m_vertices.data(), (size_t)buffer_size);

		m_device.unmapMemory(staging_buffer_memory);

		createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_vertex_buffer, m_vertex_buffer_memory);

		copyBuffer(staging_buffer, m_vertex_buffer, buffer_size);

		m_device.destroyBuffer(staging_buffer);
		staging_buffer = nullptr;

		m_device.freeMemory(staging_buffer_memory);
		staging_buffer_memory = nullptr;
	}

	void createIndexBuffer()
	{
		assert(m_device && m_indices.size() > 0);

		vk::DeviceSize buffer_size = sizeof(m_indices[0])*m_indices.size();

		vk::Buffer staging_buffer = nullptr;
		vk::DeviceMemory staging_buffer_memory = nullptr;

		createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			staging_buffer, staging_buffer_memory);

		void* data = m_device.mapMemory(staging_buffer_memory, 0, buffer_size, (vk::MemoryMapFlags)0);

		memcpy(data, m_indices.data(), (size_t)buffer_size);

		m_device.unmapMemory(staging_buffer_memory);

		createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			m_index_buffer, m_index_buffer_memory);

		copyBuffer(staging_buffer, m_index_buffer, buffer_size);

		m_device.destroyBuffer(staging_buffer);
		staging_buffer = nullptr;

		m_device.freeMemory(staging_buffer_memory);
		staging_buffer_memory = nullptr;
	}

	void createTextureImage()
	{
		assert(m_device);

		int tex_width, tex_height, tex_channels;
		stbi_uc* pixels = stbi_load(TexturePath.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

		if (pixels == nullptr)
		{
			throw std::runtime_error("Could not load texture image.");
		}

		m_mip_levels = (uint32_t)std::floor(std::log2(std::max(tex_width, tex_height))) + 1;

		vk::DeviceSize image_size = tex_width * tex_height * 4;

		vk::Buffer staging_buffer = nullptr;
		vk::DeviceMemory staging_buffer_memory = nullptr;
		createBuffer(image_size, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			staging_buffer, staging_buffer_memory);

		void* data = m_device.mapMemory(staging_buffer_memory, 0, image_size, vk::MemoryMapFlags());
		memcpy(data, pixels, static_cast<size_t>(image_size));
		m_device.unmapMemory(staging_buffer_memory);

		stbi_image_free(pixels);

		createImage((uint32_t)tex_width, (uint32_t)tex_height, m_mip_levels, vk::SampleCountFlagBits::e1,
			vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
			vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, m_texture_image, m_texture_image_memory);

		transitionImageLayout(m_texture_image, vk::Format::eR8G8B8A8Unorm,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, m_mip_levels);

		copyBufferToImage(staging_buffer, m_texture_image, (uint32_t)tex_width, (uint32_t)tex_height);

		generateMipmaps(m_texture_image, vk::Format::eR8G8B8A8Unorm, tex_width, tex_height, m_mip_levels);

		m_device.destroyBuffer(staging_buffer);
		staging_buffer = nullptr;

		m_device.freeMemory(staging_buffer_memory);
		staging_buffer_memory = nullptr;
	}

	void generateMipmaps(vk::Image image, vk::Format format, uint32_t width, uint32_t height, uint32_t mip_levels)
	{
		vk::FormatProperties format_properies = m_physical_device.getFormatProperties(format);

		if (!(format_properies.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
		{
			throw std::runtime_error("Texture image format does not support linear blitting.");
		}

		vk::CommandBuffer command_buffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier = {};
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mip_width = width;
		int32_t mip_height = height;

		for (uint32_t i = 1; i < mip_levels; ++i)
		{
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

			command_buffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(),
				0, nullptr, 0, nullptr, 1, &barrier);

			vk::ImageBlit blit = {};
			blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
			blit.srcOffsets[1] = vk::Offset3D{ mip_width, mip_height, 1 };
			blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
			blit.dstOffsets[1] = vk::Offset3D{ mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			command_buffer.blitImage(
				image, vk::ImageLayout::eTransferSrcOptimal,
				image, vk::ImageLayout::eTransferDstOptimal,
				1, &blit, vk::Filter::eLinear);

			barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
			barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			command_buffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(),
				0, nullptr, 0, nullptr, 1, &barrier);

			if (mip_width > 1) mip_width /= 2;
			if (mip_height > 1) mip_height /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mip_levels - 1;
		barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		command_buffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(),
			0, nullptr, 0, nullptr, 1, &barrier);

		endSingleTimeCommands(command_buffer);
	}

	vk::CommandBuffer beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo alloc_info = {};
		alloc_info.level = vk::CommandBufferLevel::ePrimary;
		alloc_info.commandPool = m_command_pool;
		alloc_info.commandBufferCount = 1;

		vk::CommandBuffer command_buffer = nullptr;
		m_device.allocateCommandBuffers(&alloc_info, &command_buffer);

		vk::CommandBufferBeginInfo begin_info = {};
		begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

		command_buffer.begin(begin_info);

		return command_buffer;
	}

	void endSingleTimeCommands(vk::CommandBuffer command_buffer)
	{
		command_buffer.end();

		vk::SubmitInfo submit_info = {};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		m_graphics_queue.submit(1, &submit_info, nullptr);
		m_graphics_queue.waitIdle();

		m_device.freeCommandBuffers(m_command_pool, 1, &command_buffer);
		command_buffer = nullptr;
	}

	void createImage(uint32_t width, uint32_t height,
		uint32_t mip_levels, vk::SampleCountFlagBits sample_count, vk::Format format, vk::ImageTiling tiling,
		vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties,
		vk::Image& image, vk::DeviceMemory& image_memory)
	{
		assert(m_device);

		vk::ImageCreateInfo image_info = {};
		image_info.imageType = vk::ImageType::e2D;
		image_info.extent.width = width;
		image_info.extent.height = height;
		image_info.extent.depth = 1;
		image_info.mipLevels = mip_levels;
		image_info.arrayLayers = 1;
		image_info.format = format;
		image_info.tiling = tiling;
		image_info.initialLayout = vk::ImageLayout::eUndefined;
		image_info.usage = usage;
		image_info.sharingMode = vk::SharingMode::eExclusive;
		image_info.samples = sample_count;
		image_info.flags = vk::ImageCreateFlags();

		image = m_device.createImage(image_info);

		vk::MemoryRequirements mem_requirements = m_device.getImageMemoryRequirements(image);

		vk::MemoryAllocateInfo alloc_info = {};
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

		image_memory = m_device.allocateMemory(alloc_info);

		m_device.bindImageMemory(image, image_memory, 0);
	}

	void transitionImageLayout(vk::Image image, vk::Format format,
		vk::ImageLayout old_layout, vk::ImageLayout new_layout, uint32_t mip_levels)
	{
		vk::CommandBuffer command_buffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier = {};
		barrier.oldLayout = old_layout;
		barrier.newLayout = new_layout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mip_levels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		if (new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
		{
			barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

			if (hasStencilComponent(format))
			{
				barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
			}
		}
		else
		{
			barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		}

		vk::PipelineStageFlags source_stage = {};
		vk::PipelineStageFlags destination_stage = {};

		if (old_layout == vk::ImageLayout::eUndefined &&
			new_layout == vk::ImageLayout::eTransferDstOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

			source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
			destination_stage = vk::PipelineStageFlagBits::eTransfer;
		}
		else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
			new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			source_stage = vk::PipelineStageFlagBits::eTransfer;
			destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else if (old_layout == vk::ImageLayout::eUndefined &&
			new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
				vk::AccessFlagBits::eDepthStencilAttachmentWrite;

			source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
			destination_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
		}
		else if (old_layout == vk::ImageLayout::eUndefined &&
			new_layout == vk::ImageLayout::eColorAttachmentOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
				vk::AccessFlagBits::eColorAttachmentWrite;

			source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
			destination_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		}
		else
		{
			throw std::runtime_error("Unsupported layout transition.");
		}

		command_buffer.pipelineBarrier(source_stage, destination_stage,
			vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);

		endSingleTimeCommands(command_buffer);
		command_buffer = nullptr;
	}

	void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
	{
		vk::BufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = vk::Offset3D{ 0, 0, 0 };
		region.imageExtent = vk::Extent3D{ width, height, 1 };

		vk::CommandBuffer command_buffer = beginSingleTimeCommands();

		command_buffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

		endSingleTimeCommands(command_buffer);
		command_buffer = nullptr;
	}

	void copyBuffer(vk::Buffer src_buffer, vk::Buffer dst_buffer, vk::DeviceSize size)
	{
		vk::BufferCopy copy_region = {};
		copy_region.size = size;

		vk::CommandBuffer command_buffer = beginSingleTimeCommands();
		command_buffer.copyBuffer(src_buffer, dst_buffer, 1, &copy_region);
		endSingleTimeCommands(command_buffer);
		command_buffer = nullptr;
	}

	uint32_t findMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties)
	{
		assert(m_physical_device);

		vk::PhysicalDeviceMemoryProperties mem_properties = m_physical_device.getMemoryProperties();

		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i)
		{
			if (type_filter & (1 << i) &&
				(mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type.");
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
		vk::Buffer& buffer, vk::DeviceMemory& memory)
	{
		assert(m_device);

		vk::BufferCreateInfo buffer_info = {};
		buffer_info.size = size;
		buffer_info.usage = usage;
		buffer_info.sharingMode = vk::SharingMode::eExclusive;

		buffer = m_device.createBuffer(buffer_info);

		vk::MemoryRequirements mem_requirements = m_device.getBufferMemoryRequirements(buffer);

		vk::MemoryAllocateInfo alloc_info = {};
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

		memory = m_device.allocateMemory(alloc_info);

		m_device.bindBufferMemory(buffer, memory, 0);
	}

	void createDescriptorSetLayout()
	{
		assert(m_device);

		vk::DescriptorSetLayoutBinding ubo_layout_binding = {};
		ubo_layout_binding.binding = 0;
		ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
		ubo_layout_binding.descriptorCount = 1;
		ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;
		ubo_layout_binding.pImmutableSamplers = nullptr;

		vk::DescriptorSetLayoutBinding sampler_layout_binding = {};
		sampler_layout_binding.binding = 1;
		sampler_layout_binding.descriptorCount = 1;
		sampler_layout_binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		sampler_layout_binding.stageFlags = vk::ShaderStageFlagBits::eFragment;
		sampler_layout_binding.pImmutableSamplers = nullptr;

		std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { ubo_layout_binding, sampler_layout_binding };

		vk::DescriptorSetLayoutCreateInfo layout_info = {};
		layout_info.bindingCount = (uint32_t)bindings.size();
		layout_info.pBindings = bindings.data();

		m_descriptor_set_layout = m_device.createDescriptorSetLayout(layout_info);
	}

	void createUniformBuffers()
	{
		vk::DeviceSize buffer_size = sizeof(UniformBufferObject);

		m_uniform_buffers.resize(m_swapchain_images.size());
		m_uniform_buffer_memories.resize(m_swapchain_images.size());

		for (size_t i = 0; i < m_swapchain_images.size(); ++i)
		{
			createBuffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				m_uniform_buffers[i], m_uniform_buffer_memories[i]);
		}
	}

	void createDescriptorPool()
	{
		assert(m_device);

		std::array<vk::DescriptorPoolSize, 2> pool_sizes = {};
		pool_sizes[0].type = vk::DescriptorType::eUniformBuffer;
		pool_sizes[0].descriptorCount = (uint32_t)m_swapchain_images.size();

		pool_sizes[1].type = vk::DescriptorType::eCombinedImageSampler;
		pool_sizes[1].descriptorCount = (uint32_t)m_swapchain_images.size();

		vk::DescriptorPoolCreateInfo pool_info = {};
		pool_info.poolSizeCount = (uint32_t)pool_sizes.size();
		pool_info.pPoolSizes = pool_sizes.data();
		pool_info.maxSets = (uint32_t)m_swapchain_images.size();

		m_descriptor_pool = m_device.createDescriptorPool(pool_info);
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(m_swapchain_images.size(), m_descriptor_set_layout);

		vk::DescriptorSetAllocateInfo alloc_info = {};
		alloc_info.descriptorPool = m_descriptor_pool;
		alloc_info.descriptorSetCount = static_cast<uint32_t>(m_swapchain_images.size());
		alloc_info.pSetLayouts = layouts.data();

		m_descriptor_sets = m_device.allocateDescriptorSets(alloc_info);

		for (size_t i = 0; i < m_swapchain_images.size(); ++i)
		{
			vk::DescriptorBufferInfo buffer_info = {};
			buffer_info.buffer = m_uniform_buffers[i];
			buffer_info.offset = 0;
			buffer_info.range = sizeof(UniformBufferObject);

			vk::DescriptorImageInfo image_info = {};
			image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			image_info.imageView = m_texture_image_view;
			image_info.sampler = m_texture_sampler;

			std::array<vk::WriteDescriptorSet, 2> descriptor_writes = {};
			descriptor_writes[0].dstSet = m_descriptor_sets[i];
			descriptor_writes[0].dstBinding = 0;
			descriptor_writes[0].dstArrayElement = 0;
			descriptor_writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
			descriptor_writes[0].descriptorCount = 1;
			descriptor_writes[0].pBufferInfo = &buffer_info;

			descriptor_writes[1].dstSet = m_descriptor_sets[i];
			descriptor_writes[1].dstBinding = 1;
			descriptor_writes[1].dstArrayElement = 0;
			descriptor_writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
			descriptor_writes[1].descriptorCount = 1;
			descriptor_writes[1].pImageInfo = &image_info;

			m_device.updateDescriptorSets((uint32_t)descriptor_writes.size(), descriptor_writes.data(), 0, nullptr);
		}
	}

	void createTextureImageView()
	{
		m_texture_image_view =
			createImageView(m_texture_image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, m_mip_levels);
	}

	vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspect_flags,
		uint32_t mip_levels)
	{
		vk::ImageViewCreateInfo view_info = {};
		view_info.image = image;
		view_info.viewType = vk::ImageViewType::e2D;
		view_info.format = format;
		view_info.subresourceRange.aspectMask = aspect_flags;
		view_info.subresourceRange.baseMipLevel = 0;
		view_info.subresourceRange.levelCount = mip_levels;
		view_info.subresourceRange.baseArrayLayer = 0;
		view_info.subresourceRange.layerCount = 1;

		vk::ImageView image_view = m_device.createImageView(view_info);

		return image_view;
	}

	void createTextureSampler()
	{
		vk::SamplerCreateInfo sampler_info = {};
		sampler_info.magFilter = vk::Filter::eLinear;
		sampler_info.minFilter = vk::Filter::eLinear;
		sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
		sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
		sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
		sampler_info.anisotropyEnable = VK_TRUE;
		sampler_info.maxAnisotropy = 16;
		sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
		sampler_info.unnormalizedCoordinates = VK_FALSE;
		sampler_info.compareEnable = VK_FALSE;
		sampler_info.compareOp = vk::CompareOp::eAlways;
		sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler_info.mipLodBias = 0.0f;
		sampler_info.minLod = 0.0f;
		sampler_info.maxLod = (float)m_mip_levels;

		m_texture_sampler = m_device.createSampler(sampler_info);
	}

	void createDepthResources()
	{
		vk::Format depth_format = findDepthFormat();

		createImage(m_swapchain_extent.width, m_swapchain_extent.height, 1,
			m_msaa_samples, depth_format,
			vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal, m_depth_image, m_depth_image_memory);

		m_depth_image_view = createImageView(m_depth_image, depth_format, vk::ImageAspectFlagBits::eDepth, 1);

		transitionImageLayout(m_depth_image, depth_format,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
	}

	vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling,
		vk::FormatFeatureFlags features)
	{
		for (vk::Format format : candidates)
		{
			vk::FormatProperties props = m_physical_device.getFormatProperties(format);

			if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}

		throw std::runtime_error("Failed to find supported format.");
	}

	vk::Format findDepthFormat()
	{
		return findSupportedFormat(
			{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
			vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
	}

	bool hasStencilComponent(vk::Format format)
	{
		return (format == vk::Format::eD32SfloatS8Uint) || (format == vk::Format::eD24UnormS8Uint);
	}

	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, ModelPath.c_str()))
		{
			throw std::runtime_error(err);
		}

		std::unordered_map<Vertex, uint32_t> unique_vertices;

		for (const auto& shape : shapes)
		{
			for (const auto& index : shape.mesh.indices)
			{
				Vertex vertex = {};

				vertex.pos =
				{
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.texcoord =
				{
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0 - attrib.texcoords[2 * index.texcoord_index + 1],
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				if (unique_vertices.count(vertex) == 0)
				{
					unique_vertices[vertex] = (uint32_t)m_vertices.size();
					m_vertices.push_back(vertex);
				}

				m_indices.push_back(unique_vertices[vertex]);
			}
		}
	}

	vk::SampleCountFlagBits getMaxUsableSampleCount()
	{
		vk::PhysicalDeviceProperties properties = m_physical_device.getProperties();

		vk::SampleCountFlags counts =
			(vk::SampleCountFlags)MinValue(
				(uint32_t)properties.limits.framebufferColorSampleCounts,
				(uint32_t)properties.limits.framebufferDepthSampleCounts);

		if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
		if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
		if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
		if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
		if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
		if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

		return vk::SampleCountFlagBits::e1;
	}

	void createColorResources()
	{
		vk::Format color_format = m_swapchain_image_format;

		createImage(m_swapchain_extent.width, m_swapchain_extent.height, 1, m_msaa_samples,
			color_format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal, m_color_image, m_color_image_memory);

		m_color_image_view = createImageView(m_color_image, color_format, vk::ImageAspectFlagBits::eColor, 1);

		transitionImageLayout(m_color_image, color_format,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1);
	}

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapchain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createFramebuffers();
		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSemaphoresAndFences();
	}

	void updateUniformBuffer(uint32_t image_index)
	{
		static auto start_time = std::chrono::high_resolution_clock::now();

		auto current_time = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

		float aspect = m_swapchain_extent.width / (float)m_swapchain_extent.height;

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);

		ubo.proj[1][1] *= -1.0f;

		void* data = m_device.mapMemory(m_uniform_buffer_memories[image_index], 0, sizeof(ubo), vk::MemoryMapFlags());
		memcpy(data, &ubo, sizeof(ubo));
		m_device.unmapMemory(m_uniform_buffer_memories[image_index]);
	}

	void drawFrame()
	{
		m_device.waitForFences(1, &m_in_flight_fences[m_current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());

		uint32_t image_index = 0;
		
		vk::Result result = m_device.acquireNextImageKHR(m_swapchain, std::numeric_limits<uint64_t>::max(), m_image_available_semaphores[m_current_frame], nullptr, &image_index);

		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapchain();
			return;
		}

		updateUniformBuffer(image_index);

		vk::Semaphore wait_semaphores[] = { m_image_available_semaphores[m_current_frame] };
		vk::PipelineStageFlags wait_stages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		
		vk::SubmitInfo submit_info = {};
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &m_command_buffers[image_index];

		vk::Semaphore signal_semaphores[] = { m_render_finished_semaphores[m_current_frame] };
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		m_device.resetFences(1, &m_in_flight_fences[m_current_frame]);

		m_graphics_queue.submit(1, &submit_info, m_in_flight_fences[m_current_frame]);

		vk::PresentInfoKHR present_info = {};
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		vk::SwapchainKHR swapchains[] = { m_swapchain };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapchains;
		present_info.pImageIndices = &image_index;
		present_info.pResults = nullptr;

		result = m_present_queue.presentKHR(present_info);

		if (result == vk::Result::eErrorOutOfDateKHR ||
			  result == vk::Result::eSuboptimalKHR ||
			  m_framebuffer_resized)
		{
			m_framebuffer_resized = false;
			recreateSwapchain();
		}

		m_current_frame = (m_current_frame + 1) % MaxFramesInFlight;
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(m_window))
		{
			glfwPollEvents();
			drawFrame();
		}

		m_device.waitIdle();
	}

	void cleanupSwapchain()
	{
		assert(m_device);

		m_device.destroyImageView(m_color_image_view);
		m_color_image_view = nullptr;

		m_device.destroyImage(m_color_image);
		m_color_image = nullptr;

		m_device.freeMemory(m_color_image_memory);
		m_color_image_memory = nullptr;

		m_device.destroyImageView(m_depth_image_view);
		m_depth_image_view = nullptr;

		m_device.destroyImage(m_depth_image);
		m_depth_image = nullptr;

		m_device.freeMemory(m_depth_image_memory);
		m_depth_image_memory = nullptr;

		for (auto& framebuffer : m_swapchain_frame_buffers)
		{
			m_device.destroyFramebuffer(framebuffer);
			framebuffer = nullptr;
		}

		m_device.freeCommandBuffers(m_command_pool, m_command_buffers);

		m_device.destroyPipeline(m_pipeline);
		m_pipeline = nullptr;

		m_device.destroyPipelineLayout(m_pipeline_layout);
		m_pipeline_layout = nullptr;

		m_device.destroyRenderPass(m_render_pass);
		m_render_pass = nullptr;

		for (auto& image_view : m_swapchain_image_views)
		{
			m_device.destroyImageView(image_view);
			image_view = nullptr;
		}

		m_device.destroySwapchainKHR(m_swapchain);
		m_swapchain = nullptr;
	}

	void cleanup()
	{
		for (int i = 0; i < MaxFramesInFlight; ++i)
		{
			m_device.destroySemaphore(m_image_available_semaphores[i]);
			m_image_available_semaphores[i] = nullptr;

			m_device.destroySemaphore(m_render_finished_semaphores[i]);
			m_render_finished_semaphores[i] = nullptr;

			m_device.destroyFence(m_in_flight_fences[i]);
			m_in_flight_fences[i] = nullptr;
		}

		cleanupSwapchain();

		m_device.destroySampler(m_texture_sampler);
		m_texture_sampler = nullptr;

		m_device.destroyImageView(m_texture_image_view);
		m_texture_image_view = nullptr;

		m_device.destroyImage(m_texture_image);
		m_texture_image = nullptr;

		m_device.freeMemory(m_texture_image_memory);
		m_texture_image_memory = nullptr;

		m_device.destroyDescriptorPool(m_descriptor_pool);
		m_descriptor_pool = nullptr;

		m_device.destroyDescriptorSetLayout(m_descriptor_set_layout);
		m_descriptor_set_layout = nullptr;

		for (size_t i = 0; i < m_swapchain_images.size(); ++i)
		{
			m_device.destroyBuffer(m_uniform_buffers[i]);
			m_uniform_buffers[i] = nullptr;

			m_device.freeMemory(m_uniform_buffer_memories[i]);
			m_uniform_buffer_memories[i] = nullptr;
		}

		m_device.destroyBuffer(m_index_buffer);
		m_index_buffer = nullptr;

		m_device.freeMemory(m_index_buffer_memory);
		m_index_buffer_memory = nullptr;

		m_device.destroyBuffer(m_vertex_buffer);
		m_vertex_buffer = nullptr;

		m_device.freeMemory(m_vertex_buffer_memory);
		m_vertex_buffer_memory = nullptr;

		m_device.destroyCommandPool(m_command_pool);
		m_command_pool = nullptr;

		m_device.destroy();
		m_device = nullptr;

		m_instance.destroySurfaceKHR(m_surface);
		m_surface = nullptr;

		m_instance.destroy();
		m_instance = nullptr;

		glfwDestroyWindow(m_window);
		m_window = nullptr;

		glfwTerminate();
	}

	GLFWwindow* m_window = nullptr;

	vk::Instance       m_instance        = nullptr;
	vk::PhysicalDevice m_physical_device = nullptr;
	vk::Device         m_device          = nullptr;
	vk::Queue          m_graphics_queue  = nullptr;
	vk::Queue          m_present_queue   = nullptr;

	std::vector<QueueInfo> m_queue_infos;
	
	vk::SurfaceKHR   m_surface   = nullptr;
	vk::SwapchainKHR m_swapchain = nullptr;

	std::vector<vk::Image>       m_swapchain_images;
	std::vector<vk::ImageView>   m_swapchain_image_views;
	std::vector<vk::Framebuffer> m_swapchain_frame_buffers;

	vk::Format   m_swapchain_image_format = {};
	vk::Extent2D m_swapchain_extent       = {};

	vk::RenderPass          m_render_pass           = nullptr;
	vk::DescriptorSetLayout m_descriptor_set_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout       = nullptr;
	vk::Pipeline            m_pipeline              = nullptr;

	vk::CommandPool m_command_pool = nullptr;
	std::vector<vk::CommandBuffer> m_command_buffers;

	vk::DescriptorPool m_descriptor_pool = nullptr;
	std::vector<vk::DescriptorSet> m_descriptor_sets;

	std::vector<vk::Semaphore> m_image_available_semaphores;
	std::vector<vk::Semaphore> m_render_finished_semaphores;
	std::vector<vk::Fence> m_in_flight_fences;

	size_t m_current_frame = 0;

	bool m_framebuffer_resized = false;

	std::vector<Vertex> m_vertices;
	std::vector<uint32_t> m_indices;

	vk::Buffer       m_vertex_buffer        = nullptr;
	vk::DeviceMemory m_vertex_buffer_memory = nullptr;
	vk::Buffer       m_index_buffer         = nullptr;
	vk::DeviceMemory m_index_buffer_memory  = nullptr;

	std::vector<vk::Buffer>       m_uniform_buffers;
	std::vector<vk::DeviceMemory> m_uniform_buffer_memories;

	uint32_t m_mip_levels = 1;
	vk::Image        m_texture_image = nullptr;
	vk::DeviceMemory m_texture_image_memory = nullptr;

	vk::ImageView m_texture_image_view = nullptr;
	vk::Sampler   m_texture_sampler    = nullptr;

	vk::Image        m_depth_image        = nullptr;
	vk::DeviceMemory m_depth_image_memory = nullptr;
	vk::ImageView    m_depth_image_view   = nullptr;

	vk::Image        m_color_image = nullptr;
	vk::DeviceMemory m_color_image_memory = nullptr;
	vk::ImageView    m_color_image_view = nullptr;

	vk::SampleCountFlagBits m_msaa_samples = vk::SampleCountFlagBits::e1;

	const int WindowWidth = 1920;
	const int WindowHeight = 1080;
	const int MaxFramesInFlight = 2;

	const std::string ModelPath = "models/chalet.obj";
	const std::string TexturePath = "textures/chalet.jpg";
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

