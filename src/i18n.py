"""
Internationalization support for Universal Log LUT Workflow GUI
"""

import os
import locale

SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "中文",
}

TRANSLATIONS = {
    "en": {
        # Window
        "window_title": "Universal Log LUT Workflow",
        "theme_label": "Theme:",
        "language_label": "Language:",
        "status_ready": "Ready",
        "status_processing": "Processing...",
        "status_completed": "Operation completed successfully",
        "status_error": "Error: {}",
        "error_title": "Error",
        # Tabs
        "tab_generate": "Generate LUT",
        "tab_concatenate": "Concatenate LUTs",
        "tab_compare": "Compare Images",
        "tab_resize": "Resize LUT",
        # Generate tab
        "mode": "Mode",
        "single_conversion": "Single Conversion",
        "batch_conversion": "Batch Conversion",
        "source_log_format": "Source Log Format",
        "source": "Source:",
        "target_log_format": "Target Log Format",
        "target": "Target:",
        "lut_size": "LUT Size",
        "grid_size": "Grid Size:",
        "output": "Output",
        "output_directory": "Output Directory:",
        "browse": "Browse",
        "btn_generate_lut": "Generate LUT",
        "output_log": "Output Log",
        # Concatenate tab
        "first_input_applied_first": "First Input (Applied First)",
        "file": "File",
        "directory": "Directory",
        "second_input_applied_second": "Second Input (Applied Second)",
        "output_path": "Output Path:",
        "parallel_workers": "Parallel Workers:",
        "btn_concatenate_luts": "Concatenate LUTs",
        "results": "Results",
        "col_name": "Name",
        "col_status": "Status",
        "col_clipped": "Clipped",
        "col_clip_ratio": "Clip %",
        "col_output": "Output",
        # Compare tab
        "single_image_pair": "Single Image Pair",
        "directory_comparison": "Directory Comparison",
        "first_input": "First Input",
        "second_input": "Second Input",
        "options": "Options",
        "generate_visualization": "Generate Visualization",
        "amplification": "Amplification:",
        "workers": "Workers:",
        "btn_compare_images": "Compare Images",
        # Resize tab
        "input_lut": "Input LUT",
        "target_size": "Target Size",
        "new_grid_size": "New Grid Size:",
        "output_file": "Output File:",
        "auto_generate_hint": "(Leave empty to auto-generate)",
        "btn_resize_lut": "Resize LUT",
        # Error messages
        "error_same_source_target": "Source and target cannot be the same",
        "error_specify_both_inputs": "Please specify both inputs",
        "error_specify_output": "Please specify output path",
        "error_specify_input": "Please specify input file",
        # Theme
        "theme_changed": "Theme changed to: {}",
        "theme_error_title": "Theme Error",
        "theme_error_msg": "Could not apply theme '{}': {}",
    },
    "zh": {
        # 窗口
        "window_title": "通用 Log LUT 工作流",
        "theme_label": "主题：",
        "language_label": "语言：",
        "status_ready": "就绪",
        "status_processing": "处理中...",
        "status_completed": "操作成功完成",
        "status_error": "错误：{}",
        "error_title": "错误",
        # 标签页
        "tab_generate": "生成 LUT",
        "tab_concatenate": "拼接 LUT",
        "tab_compare": "图像对比",
        "tab_resize": "调整 LUT 大小",
        # 生成标签页
        "mode": "模式",
        "single_conversion": "单个转换",
        "batch_conversion": "批量转换",
        "source_log_format": "源 Log 格式",
        "source": "源格式：",
        "target_log_format": "目标 Log 格式",
        "target": "目标格式：",
        "lut_size": "LUT 大小",
        "grid_size": "网格大小：",
        "output": "输出",
        "output_directory": "输出目录：",
        "browse": "浏览",
        "btn_generate_lut": "生成 LUT",
        "output_log": "输出日志",
        # 拼接标签页
        "first_input_applied_first": "第一个输入（先应用）",
        "file": "文件",
        "directory": "目录",
        "second_input_applied_second": "第二个输入（后应用）",
        "output_path": "输出路径：",
        "parallel_workers": "并行线程数：",
        "btn_concatenate_luts": "拼接 LUT",
        "results": "结果",
        "col_name": "名称",
        "col_status": "状态",
        "col_clipped": "裁切",
        "col_clip_ratio": "裁切比例",
        "col_output": "输出",
        # 对比标签页
        "single_image_pair": "单对图像",
        "directory_comparison": "目录对比",
        "first_input": "第一个输入",
        "second_input": "第二个输入",
        "options": "选项",
        "generate_visualization": "生成可视化",
        "amplification": "放大倍数：",
        "workers": "线程数：",
        "btn_compare_images": "图像对比",
        # 调整大小标签页
        "input_lut": "输入 LUT",
        "target_size": "目标大小",
        "new_grid_size": "新网格大小：",
        "output_file": "输出文件：",
        "auto_generate_hint": "（留空则自动生成）",
        "btn_resize_lut": "调整 LUT 大小",
        # 错误消息
        "error_same_source_target": "源格式和目标格式不能相同",
        "error_specify_both_inputs": "请指定两个输入",
        "error_specify_output": "请指定输出路径",
        "error_specify_input": "请指定输入文件",
        # 主题
        "theme_changed": "主题已切换为：{}",
        "theme_error_title": "主题错误",
        "theme_error_msg": "无法应用主题 '{}'：{}",
    },
}


def detect_language():
    """Detect OS language, return 'zh' for Chinese, 'en' otherwise"""
    for env_var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        lang = os.environ.get(env_var, "")
        if lang.lower().startswith("zh"):
            return "zh"
    try:
        loc = locale.getlocale()
        if loc and loc[0] and loc[0].lower().startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"
