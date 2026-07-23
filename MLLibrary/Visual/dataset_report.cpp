#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    constexpr std::size_t kImageWidth = 28;
    constexpr std::size_t kImageHeight = 28;
    constexpr std::size_t kPixelsPerImage = kImageWidth * kImageHeight;
    constexpr std::size_t kClassCount = 10;

    bool ReadFloatFile(const char* path, std::vector<float>& values)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) return false;
        const std::streamsize byteSize = input.tellg();
        if (byteSize <= 0 || byteSize % static_cast<std::streamsize>(sizeof(float)) != 0) return false;
        values.resize(static_cast<std::size_t>(byteSize) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(values.data()), byteSize);
        return input.good();
    }

    bool ParseSampleLimit(std::string_view text, std::size_t& value)
    {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc{} && end == text.data() + text.size() && value > 0 && value <= 500;
    }

    void WriteNumberArray(std::ostream& output, std::span<const float> values)
    {
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index != 0) output << ',';
            output << std::clamp(values[index], 0.0f, 1.0f);
        }
    }
}

int main(int argumentCount, char** arguments)
{
    if (argumentCount < 4 || argumentCount > 5)
    {
        std::cerr << "Usage: " << arguments[0] << " <images.mat> <labels.mat> <report.html> [sample-limit]\n";
        return 2;
    }

    std::size_t sampleLimit = 64;
    if (argumentCount == 5 && !ParseSampleLimit(arguments[4], sampleLimit))
    {
        std::cerr << "Sample limit must be an integer from 1 through 500.\n";
        return 2;
    }

    std::vector<float> images;
    std::vector<float> labels;
    if (!ReadFloatFile(arguments[1], images) || !ReadFloatFile(arguments[2], labels))
    {
        std::cerr << "Image and label files must be non-empty raw Float32 files.\n";
        return 1;
    }
    if (images.size() != labels.size() * kPixelsPerImage)
    {
        std::cerr << "Expected exactly 784 Float32 pixels for each label.\n";
        return 1;
    }

    std::array<std::size_t, kClassCount> classCounts{};
    double pixelSum = 0.0;
    double pixelSquareSum = 0.0;
    float pixelMinimum = std::numeric_limits<float>::infinity();
    float pixelMaximum = -std::numeric_limits<float>::infinity();
    for (float pixel : images)
    {
        if (!std::isfinite(pixel))
        {
            std::cerr << "Image data contains a non-finite value.\n";
            return 1;
        }
        pixelSum += pixel;
        pixelSquareSum += static_cast<double>(pixel) * pixel;
        pixelMinimum = std::min(pixelMinimum, pixel);
        pixelMaximum = std::max(pixelMaximum, pixel);
    }
    for (float label : labels)
    {
        if (!std::isfinite(label) || label < 0.0f || label >= static_cast<float>(kClassCount)
            || std::floor(label) != label)
        {
            std::cerr << "Labels must be integral Float32 values from 0 through 9.\n";
            return 1;
        }
        ++classCounts[static_cast<std::size_t>(label)];
    }

    const double mean = pixelSum / static_cast<double>(images.size());
    const double variance = std::max(0.0, pixelSquareSum / static_cast<double>(images.size()) - mean * mean);
    sampleLimit = std::min(sampleLimit, labels.size());
    std::ofstream output(arguments[3]);
    if (!output)
    {
        std::cerr << "Cannot write report: " << arguments[3] << '\n';
        return 1;
    }

    output << R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>MLLibrary Dataset Report</title><style>
body{margin:0;background:#f4f6f8;color:#18212f;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:24px;margin:0 0 6px}h2{font-size:16px;margin:0 0 14px}.muted{color:#667386}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:#d8dee6;border:1px solid #d8dee6;margin:24px 0}.metric{background:#fff;padding:15px}.metric strong{display:block;font-size:22px;margin-top:5px}section{margin-top:28px}.histogram{display:grid;grid-template-columns:repeat(10,1fr);gap:10px;height:190px;align-items:end}.bar{height:var(--height);min-height:2px;background:#2e6f95;position:relative}.bar span{position:absolute;bottom:calc(100% + 5px);font-size:11px}.bar b{position:absolute;top:calc(100% + 5px);left:50%;transform:translateX(-50%)}canvas{display:block;width:100%;height:auto;background:#fff;border:1px solid #d8dee6;image-rendering:pixelated}@media(max-width:600px){.metric:last-child{grid-column:1/-1}.histogram{gap:6px}}</style></head><body><main><h1>Dataset Report</h1><div class="muted">Validated raw Float32 MNIST-compatible dataset</div><div class="metrics"><div class="metric"><span class="muted">Samples</span><strong>)HTML"
           << labels.size() << R"HTML(</strong></div><div class="metric"><span class="muted">Shape</span><strong>28 × 28</strong></div><div class="metric"><span class="muted">Pixel mean</span><strong>)HTML"
           << mean << R"HTML(</strong></div><div class="metric"><span class="muted">Pixel stddev</span><strong>)HTML"
           << std::sqrt(variance) << R"HTML(</strong></div><div class="metric"><span class="muted">Pixel range</span><strong>)HTML"
           << pixelMinimum << " .. " << pixelMaximum << R"HTML(</strong></div></div><section><h2>Class distribution</h2><div class="histogram">)HTML";

    const std::size_t largestClass = *std::max_element(classCounts.begin(), classCounts.end());
    for (std::size_t category = 0; category < classCounts.size(); ++category)
    {
        const double percent = largestClass == 0 ? 0.0 : 100.0 * classCounts[category] / largestClass;
        output << "<div class=\"bar\" style=\"--height:" << percent << "%\"><span>" << classCounts[category]
               << "</span><b>" << category << "</b></div>";
    }
    output << R"HTML(</div></section><section><h2>Deterministic sample grid</h2><canvas id="samples"></canvas></section><script>
const width=28,height=28,scale=4,padding=20;
const labels=[)HTML";
    for (std::size_t sample = 0; sample < sampleLimit; ++sample)
    {
        if (sample != 0) output << ',';
        output << static_cast<unsigned int>(labels[sample]);
    }
    output << R"HTML(],images=[)HTML";
    for (std::size_t sample = 0; sample < sampleLimit; ++sample)
    {
        if (sample != 0) output << ',';
        output << '[';
        WriteNumberArray(output, std::span(images.data() + sample * kPixelsPerImage, kPixelsPerImage));
        output << ']';
    }
    output << R"HTML(];const c=document.getElementById("samples");function render(){const columns=innerWidth<600?4:8,rows=Math.ceil(images.length/columns),cellW=width*scale+padding,cellH=height*scale+padding+8,ratio=devicePixelRatio||1;c.width=columns*cellW*ratio;c.height=rows*cellH*ratio;c.style.aspectRatio=`${columns*cellW}/${rows*cellH}`;const x=c.getContext("2d");x.scale(ratio,ratio);x.fillStyle="#fff";x.fillRect(0,0,columns*cellW,rows*cellH);images.forEach((pixels,index)=>{const ox=(index%columns)*cellW+padding/2,oy=Math.floor(index/columns)*cellH+padding/2,img=x.createImageData(width,height);pixels.forEach((value,pixel)=>{const shade=Math.round(255*(1-value)),offset=pixel*4;img.data[offset]=shade;img.data[offset+1]=shade;img.data[offset+2]=shade;img.data[offset+3]=255});const tile=document.createElement("canvas");tile.width=width;tile.height=height;tile.getContext("2d").putImageData(img,0,0);x.imageSmoothingEnabled=false;x.drawImage(tile,ox,oy,width*scale,height*scale);x.fillStyle="#18212f";x.font="12px sans-serif";x.fillText(`label ${labels[index]}`,ox,oy+height*scale+16)})}render();addEventListener("resize",render);</script></main></body></html>)HTML";
    return output ? 0 : 1;
}
