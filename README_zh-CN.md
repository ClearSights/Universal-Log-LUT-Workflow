# **电影感调色通用方案：适用于任何相机的 Log LUT 工作流**

[简体中文](README_zh-CN.md) | [English](README.md)

# **1\. 概述**

本指南介绍一套强大的工作流程，可将高质量、符合行业标准的 3D LUT（例如 ARRI 的 Look Library 或富士胶片的 Film Simulations）应用到**任何** RAW 照片文件，无论相机品牌。

通过将 RAW 文件转换为标准化的“电影级”中间格式（本指南选用 Arri LogC4 作为中间格式），我们将相机传感器与后期处理流程解耦。这使得您可以在 Adobe Lightroom 和 Photoshop 中，对来自所有品牌（如 Sony、Canon、Nikon 等）的素材使用统一的专业色彩校正工具。

# **2\. 核心概念**

在深入操作步骤前，理解以下三项核心概念有助于您更好地掌握这套工作流。

- **Log 曲线：** 标准 JPEG 图像内置了高对比度曲线。而“Log”图像则显得灰平、饱和度低，这是因为它们在阴影和高光区域保留了更多数据，类似于数字底片。电影摄影机（如 ARRI Alexa）采用“Log”格式记录，正是为了捕捉最大的动态范围。
- **色域 (gamut):** 指文件能够显示的颜色范围。
  - _Rec.709:_ 普通显示器的标准色彩空间（范围较小）。
  - _ARRI Wide Gamut (AWG):_ 电影工业使用的大范围色彩空间。我们选择它作为中间色彩空间，是因为它足够大，能够容纳几乎所有现代相机传感器的色彩信息而不会发生裁剪。
- **3D LUT：** 您可以将 LUT 理解为一个复杂的色彩“翻译词典”。它接收一种输入色彩（例如，“灰平的 Log 红色”），并将其转换为一种输出色彩（例如，“鲜艳的电影红色”）。大多数专业 LUT 都期望接收特定格式的 Log 输入（如 LogC4）。如果直接对标准 RAW 照片应用这些 LUT，色彩会显得怪异。本工作流正是为了解决这种不匹配问题。

# **3\. 方案架构**

这个方案包括两个主要阶段：

1. **准备阶段：** 创建一个“通用 LUT”，将我们选定的工作空间（LogC4）转换为最终的目标“风格”。
2. **处理阶段：** 将 RAW 照片转换为 LogC4 格式的 TIFF 文件，导入 Lightroom，并通过相机配置文件应用 LUT。

**所需工具：**

- [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy/)（用于 RAW 转换）
- Adobe Photoshop
- 自定义 Python 脚本：
  - `generate_log2log_lut.py`
  - `concatenate_luts.py`

# **4\. 详细步骤**

## **阶段 1：创建“增强配置文件”（风格文件）**
由于我们的目标是将所有素材标准化至 **ARRI LogC4**，因此在 Lightroom 中使用的 LUT 必须代表这样的转换：“_将这个 LogC4 图像处理成类似富士经典正片（Fujifilm Classic Chrome）的效果_。”

### **步骤 1.1：确定 LUT 转换路径**

根据您使用的 LUT 来源，您可能需要桥接不同的色彩空间。

**场景 A：使用原生 ARRI Look Library**

- **LUT 来源：** ARRI 风格 LUT (LogC4 → LogC4) + ARRI 还原 LUT (LogC4 → Rec709)。
- **操作：** 使用`concatenate_luts.py`将它们合并成一个文件。
  - `python concatenate_luts.py -i1 logc4_to_logc4.cube -i2 logc4_to_rec709.cube -o output.cube`
- **结果：** 获得一个接收 LogC4 输入并直接输出最终 Rec709 图像的 LUT。

**场景 B：使用富士胶片或其他品牌（桥接）**

- **LUT 来源：** 富士胶片模拟 LUT（F-Log2C → Rec709）。
- **操作：**
  - 运行`generate_log2log_lut.py`创建一个“桥接 LUT”，用于转换 LogC4 → F-Log2C。
    - `python generate_log2log_lut.py --source LogC4 --target F-Log2C --size 65 --output logc4_to_flog2c.cube`
  - 运行`concatenate_luts.py`合并**桥接 LUT**与**富士胶片 LUT**。
    - `python concatenate_luts.py -i1 logc4_to_flog2c.cube -i2 flog2c_to_rec709.cube -o output.cube`
- **结果：** 获得一个 LUT，它能接收 LogC4 输入，通过数学计算将其转换为 F-Log2C，然后应用富士的“风格”，所有步骤在一个文件中完成。

### **步骤 1.2：将 LUT 转换为 Camera Raw 配置文件**

生成`.cube` 文件（例如`LogC4_Fuji_ClassicChrome.cube`）后，需要使其能在 Lightroom 中使用。

1. 使用 **Adobe Photoshop** 打开任意一张图片。
2. 按 `Ctrl \+ Shift \+ A` (Windows) 或 `Command \+ Option \+ A` (Mac) 打开 **Adobe Camera Raw (ACR)**
3. 打开右侧的 **Presets（预设）**面板
4. 按住 `Alt`** (Windows) 或 `Option` (Mac) 键，同时点击 **Create Preset（创建预设）**按钮，以打开 **Create Profile（创建配置文件）**对话框。
   - 注意：如果不按住 `Alt`/`Option` 键直接点击，会打开 **Create Preset** 对话框，这是错误的。请确保对话框标题为 **Create Profile**。
5. 按以下说明设置（可参考下方图片）：
   - **Name（名称）：** 为其设置一个清晰的名称（例如，“LogC4 - 富士经典正片”）。
   - **Group（组）：** 为其安排分组（例如，新建一个名为“通用 Log 风格”的组）。
   - **Current Image Settings to Include（包含的当前图像设置）：** **取消勾选**所有项目
   - **Advanced Settings（高级设置）：** 将“Tone Map Strength（色调映射强度）”设为“Low (Normal)（低 - 正常）”。
   - **Look Table（查找表）：** **取消勾选**
   - **Color Lookup Table（颜色查找表）：** **勾选**此项，并加载您生成的 `.cube` 文件。

![image2](./static/1.png)

**图 1.** 创建配置文件的设置参考

6. 点击“确定”。此配置文件现在即可在 Photoshop 和 Lightroom 中使用。

## **阶段 2：处理图片**

配置文件准备就绪后，即可开始处理图片。

### **步骤 2.1：将 RAW 转换为 LogC4 TIFF**

使用 Raw-Alchemy 处理您相机的 RAW 文件（如 .CR3, .ARW, .NEF 等）。

- **设置：** 将 Raw-Alchemy 配置为使用 **ARRI LogC4** 曲线和 **ARRI Wide Gamut 4** 色彩空间输出 **16 位 TIFF** 文件。
  - **Input Path（输入路径）：** 您的 RAW 文件所在文件夹。
  - **Output Path（输出路径）：** 存放输出 LogC4 文件的文件夹。
  - **Log Space（Log 空间）：** Arri LogC4
  - **LUT File (.cube)（LUT 文件）：** **留空**（除非您明确知道需要在此处插入额外的 Log-to-Log LUT）
- _注意：_ 我们选择 16 位 TIFF 是因为 8 位色深不足以承载 Log 曲线的数据，可能导致“色带”（色彩渐变中出现条带）现象。

![image2](./static/2.png)

**图 2.** 批量将 RAW 图像转换为 16 位 LogC4 TIFF 文件的设置参考

### **步骤 2.2：导入和应用**

1. 将生成的 TIFF 文件导入 **Lightroom** 或 **Photoshop**。
2. 打开 **Profile Browser**（配置文件浏览器）（位于“基本”面板中）。
3. 选择您想要的配置文件（例如，“LogC4 \- Fuji Classic Chrome”）。
4. _结果：_ 原本灰平、低饱和度的 Log 图像会立即转换为一张色彩校正完美的高质量图像。

# **5\. 验证准确性**

我们如何确认这套流程确实有效，而非仅仅是“看起来差不多”？

我们使用一个包含 50 个 RAW 文件的小型测试集（涵盖多种光照条件和色彩）进行了验证实验。

1. **参考图像：** 完全通过代码生成：使用 Raw-Alchemy 将 RAW → FLog2C → 应用 LUT `FLog2C_to_PROVIA_65grid_V.1.00.cube` → 输出。
2. **测试图像（本工作流）：** RAW → LogC4 TIFF（通过 Raw-Alchemy）→ 导入 Lightroom → 应用配置文件（LUT）→ 导出为 16 位 TIFF。

使用`compare_images.py`脚本，我们对比了参考图像和测试图像的像素值。

- **评估指标：**
  - 数值差异：平均/最大/标准差像素差异。
  - 感知差异：8 位等效平均/最大值、最小可察觉差异（JND）阈值、可察觉性等。
  - Delta E 色彩差异：平均/最大值等。
- **实验结果：**
  - 像素差异均 <= 16（在 16 位图像中，每个通道的像素值范围为 0-65535）。
  - 所有样本的 **可察觉性得分和 ΔE 均为 0**。
- **结论：**
  - 图像间的数值差异远低于人类视觉感知极限（远小于 8 位色彩中单个整数值的差异）。
  - 像素间的微小数值差异对进一步后期处理的影响可忽略不计。
  - Lightroom 的输出可以被视为与纯代码转换的结果**视觉上完全相同**。

# **6\. 优点与局限性**

## **优点**

1. **通用性强：** 您可以将完全相同的“风格”（例如，特定的柯达胶片模拟）应用到 Sony a7V、Canon R5 和 Nikon Z8 拍摄的照片上，并且它们能完美匹配，因为所有素材都在“LogC4”这个中间平台上完成了统一。
2. **提升工作流效率：** 一旦创建好配置文件，后续操作就是标准的 Lightroom 流程。您可以进行批量编辑、同步设置，并在此电影级调色的基础上，继续使用 Lightroom 和 Photoshop 强大的调整编辑功能。
3. **精度高：** 整个流程采用 16 位浮点/整数处理，有效防止了图像质量损失。
4. **扩展资源库：** 为摄影师打开了通往高端视频 LUT 世界的大门（这些 LUT 通常比普通的照片预设质量更高）。

## **局限性**

1. **占用存储空间：** 16 位 TIFF 文件比原始 RAW 文件或 JPEG 文件大得多（需要额外准备存储空间）。
2. **初始设置较复杂：** 需要一定的命令行工具（Python）使用经验来生成初始的复合 LUT。
3. **白平衡调整灵活性受限：** 由于图像被“烘焙”为 TIFF 格式，在 Lightroom 中进行极限白平衡调整时，不如直接处理原始 Bayer 数据灵活。尽管 16 位深度在很大程度上缓解了此问题，但仍建议在 Raw-Alchemy 转换阶段（如有可能）就设置好准确的白平衡。

# **7\. 部分测试图片的比较结果**

![image2](./static/3.png)

图 3. 测试图片 1 的比较结果

![image2](./static/4.png)

图 4. 测试图片 2 的比较结果

![image2](./static/5.png)

图 5. 测试图片 3 的比较结果

![image2](./static/6.png)

图 6. 测试图片 4 的比较结果
