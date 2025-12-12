# **Cinematic Color for Any Camera: The Universal Log LUT Workflow**

[English](README.md) | [简体中文](README_zh-CN.md)

# **1\. Overview**

This guide introduces a robust workflow to apply high-quality, industry-standard 3D LUTs (such as ARRI’s Look Library or Fujifilm’s Film Simulations) to **any** RAW photo file, regardless of the camera manufacturer.

By converting RAW files into a standardized "cinema-grade" intermediate format (In this guide we use Arri LogC4 as the intermediate format), we decouple the camera sensor from the editing process. This allows you to use a single set of professional color grading tools across all your cameras (whether you shoot Sony, Canon, Nikon, or others) within Adobe Lightroom and Photoshop.

# **2\. Core Concepts**

Before diving into the steps, it helps to understand three key concepts that make this workflow possible.

- **Log Curves (Logarithmic Gamma):** Standard JPEG images have high contrast baked in. "Log" images look flat and washed out because they preserve more data in the shadows and highlights, similar to a digital negative. Movie cameras (like ARRI Alexa) record in "Log" to capture maximum dynamic range.
- **Color Gamut (The Palette):** This is the range of colors a file can display.
  - _Rec.709:_ The standard for normal monitors (smaller range).
  - _ARRI Wide Gamut (AWG):_ A massive color space used in cinema. We use this because it is large enough to contain the color information from almost any modern camera sensor without clipping.
- **3D LUTs (Look-Up Tables):** Think of a LUT as a complex translation dictionary. It takes an input color (e.g., "flat log red") and translates it to an output color (e.g., "vibrant filmic red"). Most professional LUTs expect a specific Log input (like LogC4). If you feed them a standard RAW photo, the colors will look wrong. This workflow fixes that mismatch.

# **3\. The Workflow Architecture**

The process consists of two main phases:

1. **Preparation:** Creating a "Universal LUT" that translates our chosen working space (LogC4) into the final Look.
2. **Processing:** converting the RAW photo to a LogC4 TIFF, importing it into Lightroom, and applying the LUT via a Camera Profile.

**Tools Required:**

- [Raw-Alchemy](https://github.com/shenmintao/Raw-Alchemy/) (for RAW conversion)
- Adobe Photoshop
- Custom Python Scripts (provided below):
  - `generate_log2log_lut.py`
  - `concatenate_luts.py`

# **4\. Step-by-Step Implementation**

## **Phase 1: Creating the "Enhanced Profile" (The Look)**

Since our goal is to standardize everything to **ARRI LogC4**, the LUTs we use in Lightroom must effectively say: _"Take this LogC4 image and make it look like Fujifilm Classic Chrome."_

### **Step 1.1: Determine LUT Conversion Path**

Depending on the source of your LUT, you may need to bridge different color spaces.

**Scenario A: Using ARRI Look Library (Native)**

- **Source:** ARRI Look LUT (LogC4 → LogC4) \+ Output Transform (LogC4 → Rec709).
- **Action:** Use `concatenate_luts.py` to merge them into a single file.
  - `python concatenate_luts.py -i1 logc4_to_logc4.cube -i2 logc4_to_rec709.cube -o output.cube`
- **Result:** A LUT that takes LogC4 input and outputs the final Rec709 image.

**Scenario B: Using Fujifilm or Other Brands (Bridging)**

- **Source:** Fujifilm Simulation LUT (Expects F-Log2C input → Outputs Rec709).
- **Action:**
  - Run `generate_log2log_lut.py` to create a "Bridge LUT" that converts **LogC4 → F-Log2C**.
    - `python generate_log2log_lut.py --source LogC4 --target F-Log2C --size 65 --output logc4_to_flog2c.cube`
  - Run `concatenate_luts.py` to merge the **Bridge LUT** \+ **Fujifilm LUT**.
    - `python concatenate_luts.py -i1 logc4_to_flog2c.cube -i2 flog2c_to_rec709.cube -o output.cube`
- **Result:** A LUT that takes LogC4 input, mathematically transforms it to F-Log2, and applies the Fuji look, all in one file.

### **Step 1.2: Convert LUT to Camera Raw Profile**

Once you have your generated .cube file (e.g., LogC4_Fuji_ClassicChrome.cube), you need to make it usable in Lightroom.

1. Open **Adobe Photoshop** with any dummy image
2. Open **Adobe Camera Raw (ACR)** with Ctrl \+ Shift \+ A (Windows) or Command \+ Option \+ A (Mac)
3. Open the **Presets** tab on the right
4. Hold the **Alt (Windows)** or **Option (Mac)** key and click on the **Create Preset** button to access the **Create Profile** dialog
   - If the **Create Preset** button is clicked without holding the **Alt (Windows)** or **Option (Mac)** key, a **Create Preset** dialog will be open. This is the WRONG dialog. Make sure the dialog window name is **Create Profile**.
5. Settings (as shown in screenshot below):
   - **Name:** Give it a clear name (e.g., "LogC4 \- Fuji Classic Chrome").
   - **Group:** Arrange its location (e.g. create a "Universal Log Looks" group).
   - **Current Image Settings to Include:** Uncheck all
   - **Advanced Settings:** Set "Tone Map Strength" to "Low (Normal)"
   - **Look Table:** Uncheck it
   - **Color Lookup Table:** Check this box and load your generated .cube file.

![image1](./static/1.png)

**Figure 1\.** Reference setting to create profile

1. Click OK. This profile is now available in both Photoshop and Lightroom.

## **Phase 2: Processing the Image**

Now that the profile is ready, we prepare the image.

### **Step 2.1: Convert RAW to LogC4 TIFF**

Use Raw-Alchemy to process your camera's RAW files (CR3, ARW, NEF, etc.).

- **Settings:** Configure Raw-Alchemy to output **16-bit TIFF** files using the **ARRI LogC4** curve and **ARRI Wide Gamut 4** color space.
  - **Input Path:** Your folder of RAW files
  - **Output Path:** Folder for output LogC4 files
  - **Log Space:** Arri LogC4
  - **LUT File (.cube):** **LEAVE EMPTY** (Unless you know what you’re doing. Additional log-to-log LUT can be inserted here)
- _Note:_ We use 16-bit TIFF because 8-bit does not have enough data depth to handle Log curves without "banding" (visible stripes in gradients).

![image2](./static/2.png)

**Figure 2\.** Reference setting to batch convert RAW image files to 16-bit LogC4 TIFF files

### **Step 2.2: Import and Apply**

1. Import the resulting TIFF files into **Lightroom** or **Photoshop**.
2. Open the **Profile Browser** (in the Basic panel).
3. Select your desired profile (e.g., "LogC4 \- Fuji Classic Chrome").
4. _Result:_ The flat, grey Log image instantly snaps into a perfectly graded, high-quality image.

# **5\. Verification & Accuracy**

How do we know this actually works and isn't just "close enough"?

I performed a validation experiment using a small test dataset of 50 RAW files spanning various lighting conditions and colors.

1. **Reference Image:** Generated entirely in code using Raw-Alchemy, converting RAW → FLog2C → Apply LUT FLog2C_to_PROVIA_65grid_V.1.00.cube → Output.
2. **Test Image (Workflow):** RAW → LogC4 TIFF (via Raw-Alchemy) → Lightroom Import → Applied Profile (LUT) → Exported as 16-bit TIFF.

Using the script compare_images.py, I compared the pixel values between the Reference and Test images. Examples are provided at the end of this guide.

- **Metrics:**
  - Difference: mean/max/std pixel difference
  - Perceptual: 8-bit equivalent mean/max, Just Noticeable Difference (JND) threshold, perceptibility, etc.
  - Delta E: mean/max, etc.
- **Results:**
  - The pixel differences are all \<= 16 (In a 16-bit image, pixel value per channel ranges between 0 and 65535).
  - The **Perceptibility Score and ΔE are 0** for all samples.
- **Conclusion:**
  - The mathematical difference between the images is far below the human perception limit (much lower than a single integer value difference in 8-bit color).
  - The impact of the mathematical difference between pixels is minimal in further editing.
  - The Lightroom output can be considered as identical to the pure code-based conversion.

# **6\. Advantages & Limitations**

## **Advantages**

1. **Universality:** You can apply the exact same "Look" (e.g., a specific Kodak film emulation) to a Sony a7V, a Canon R5, and a Nikon Z8, and they will match perfectly because they meet in the "LogC4" middle ground.
2. **Workflow Efficiency:** Once the profiles are created, the workflow is standard Lightroom. You can batch edit, sync settings, and use Lightroom's masking tools on top of the cinema grade.
3. **High Precision:** Working in 16-bit float/int throughout the pipeline prevents image degradation.
4. **Access to Cinema Libraries:** Opens up the world of high-end video LUTs (which are often higher quality than standard photo presets) to photographers.

## **Limitations**

1. **Storage Space:** 16-bit TIFF files are significantly larger than RAW files or JPEGs. (Plan for extra disk space).
2. **Initial Setup:** Requires familiarity with command-line tools (Python) to generate the initial combined LUTs.
3. **White Balance:** Because the image is "baked" into a TIFF, extreme White Balance adjustments in Lightroom are slightly less flexible than on a raw Bayer layer, though 16-bit depth mitigates most of this issue. It is best to set WB correctly in Raw-Alchemy (if possible) during the initial conversion.

# **7\. Example Comparison Results**

![image2](./static/3.png)

Figure 3\. Comparison result of test image 1

![image2](./static/4.png)

Figure 4\. Comparison result of test image 2

![image2](./static/5.png)

Figure 5\. Comparison result of test image 3

![image2](./static/6.png)

Figure 6\. Comparison result of test image 4
