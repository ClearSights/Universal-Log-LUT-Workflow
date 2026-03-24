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
        "tab_generate": "Create LUT",
        "tab_concatenate": "Concatenate LUTs",
        "tab_compare": "Compare Images",
        "tab_resize": "Resize LUT",
        # Generate tab
        "mode": "Mode",
        "single_conversion": "Create Single LUT",
        "batch_conversion": "Batch Create LUTs(Source Format Fixed)",
        "source_log_format": "Source Log Format",
        "source": "Source:",
        "target_log_format": "Target Log Format",
        "target": "Target:",
        "lut_size": "LUT Size",
        "grid_size": "Grid Size:",
        "output": "Output",
        "output_directory": "Output Directory:",
        "browse": "Browse",
        "btn_generate_lut": "Create LUT",
        "output_log": "Operation Log",
        # Concatenate tab
        "first_input_applied_first": "First Input (Applied First)",
        "file": "File",
        "directory": "Directory",
        "second_input_applied_second": "Second Input (Applied Second)",
        "output_path": "Output Path:",
        "parallel_workers": "Parallel Workers:",
        "btn_concatenate_luts": "Concatenate",
        "results": "Results",
        "col_name": "Name",
        "col_status": "Status",
        "col_clipped": "Clipped",
        "col_clip_ratio": "Clip %",
        "col_output": "Output",
        # Compare tab
        "single_image_pair": "Single Image Pair",
        "directory_comparison": "Batch Comparison (Directory)",
        "first_input": "First Image/Directory",
        "second_input": "Second Image/Directory",
        "options": "Options",
        "generate_visualization": "Generate Visualization Comparison",
        "amplification": "Amplification:",
        "workers": "Workers:",
        "btn_compare_images": "Start Comparison",
        # Resize tab
        "input_lut": "Input LUT",
        "target_size": "Target Size",
        "new_grid_size": "New Grid Size:",
        "output_file": "Output File:",
        "auto_generate_hint": "(Leave empty to auto-generate)",
        "btn_resize_lut": "Resize",
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
        "tab_generate": "创建 LUT",
        "tab_concatenate": "拼接 LUT",
        "tab_compare": "图像对比",
        "tab_resize": "调整 LUT 规格",
        # 生成标签页
        "mode": "模式",
        "single_conversion": "创建单个 LUT",
        "batch_conversion": "批量创建 LUT (固定源格式)",
        "source_log_format": "输入 Log 格式",
        "source": "输入格式：",
        "target_log_format": "输出 Log 格式",
        "target": "输出格式：",
        "lut_size": "LUT 规格",
        "grid_size": "网格大小：",
        "output": "文件输出",
        "output_directory": "LUT 文件输出目录：",
        "browse": "浏览",
        "btn_generate_lut": "创建 LUT",
        "output_log": "操作日志",
        # 拼接标签页
        "first_input_applied_first": "第一个 LUT(s) (在前)",
        "file": "文件",
        "directory": "目录",
        "second_input_applied_second": "第二个 LUT(s) (在后)",
        "output_path": "输出路径：",
        "parallel_workers": "并行线程数：",
        "btn_concatenate_luts": "拼接",
        "results": "结果",
        "col_name": "名称",
        "col_status": "状态",
        "col_clipped": "裁切",
        "col_clip_ratio": "裁切比例",
        "col_output": "输出",
        # 对比标签页
        "single_image_pair": "单组图像对比",
        "directory_comparison": "批量对比（文件夹）",
        "first_input": "第一张/组图像",
        "second_input": "第二张/组图像",
        "options": "选项",
        "generate_visualization": "生成可视化对比图",
        "amplification": "放大倍数：",
        "workers": "并行线程数：",
        "btn_compare_images": "开始对比",
        # 调整规格标签页
        "input_lut": "输入 LUT",
        "target_size": "目标规格",
        "new_grid_size": "目标网格大小：",
        "output_file": "输出文件：",
        "auto_generate_hint": "（留空则自动生成）",
        "btn_resize_lut": "调整规格",
        # 错误消息
        "error_same_source_target": "源格式和目标格式不能相同",
        "error_specify_both_inputs": "请将两个输入都指定完整",
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
