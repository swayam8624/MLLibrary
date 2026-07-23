#include <charconv>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Metric final {
    double epoch = 0.0;
    double accuracy = 0.0;
    double loss = 0.0;
    double learningRate = 0.0;
};

bool parseNumber(std::string_view text, double& output)
{
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [pointer, error] = std::from_chars(begin, end, output);
    return error == std::errc{} && pointer == end;
}

bool parseMetric(std::string_view line, Metric& metric)
{
    std::string_view fields[4];
    std::size_t start = 0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        const std::size_t comma = line.find(',', start);
        if (index < 3 && comma == std::string_view::npos) return false;
        fields[index] = line.substr(start, comma == std::string_view::npos ? line.size() - start : comma - start);
        start = comma == std::string_view::npos ? line.size() : comma + 1;
    }
    return parseNumber(fields[0], metric.epoch) && parseNumber(fields[1], metric.accuracy)
        && parseNumber(fields[2], metric.loss) && parseNumber(fields[3], metric.learningRate);
}

void writeSeries(std::ostream& output, const std::vector<Metric>& metrics, double Metric::*member)
{
    for (std::size_t index = 0; index < metrics.size(); ++index)
    {
        if (index != 0) output << ',';
        output << metrics[index].*member;
    }
}

} // namespace

int main(int argumentCount, char** arguments)
{
    if (argumentCount != 3)
    {
        std::fprintf(stderr, "Usage: %s <training_metrics.csv> <report.html>\n", arguments[0]);
        return 2;
    }
    std::ifstream input(arguments[1]);
    if (!input)
    {
        std::fprintf(stderr, "Cannot open metrics file: %s\n", arguments[1]);
        return 1;
    }
    std::string line;
    if (!std::getline(input, line) || line != "epoch,accuracy,cost,learning_rate")
    {
        std::fputs("Metrics CSV must use the expected training header.\n", stderr);
        return 1;
    }
    std::vector<Metric> metrics;
    while (std::getline(input, line))
    {
        if (line.empty()) continue;
        Metric metric;
        if (!parseMetric(line, metric))
        {
            std::fputs("Metrics CSV contains an invalid numeric row.\n", stderr);
            return 1;
        }
        metrics.push_back(metric);
    }
    if (metrics.empty())
    {
        std::fputs("Metrics CSV contains no epoch rows.\n", stderr);
        return 1;
    }

    std::ofstream output(arguments[2]);
    if (!output)
    {
        std::fprintf(stderr, "Cannot write report: %s\n", arguments[2]);
        return 1;
    }
    const Metric& final = metrics.back();
    output << R"HTML(<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLLibrary Training Report</title><style>
body{margin:0;background:#f5f7fa;color:#18212f;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:0 auto;padding:32px}.summary{display:flex;gap:24px;border-bottom:1px solid #dce1e8;padding-bottom:24px}.metric{min-width:160px}.metric b{display:block;font-size:25px;margin-top:5px}section{margin-top:28px;background:#fff;border:1px solid #dce1e8;border-radius:6px;padding:18px}canvas{width:100%;height:280px;display:block}h1{margin:0;font-size:24px}h2{font-size:16px;margin:0 0 12px}small{color:#637083}</style></head><body><main><h1>Training Report</h1><p><small>Generated locally from MLLibrary CSV metrics.</small></p><div class="summary"><div class="metric"><small>Epochs</small><b>)HTML" << metrics.size() << R"HTML(</b></div><div class="metric"><small>Final accuracy</small><b>)HTML" << final.accuracy * 100.0 << R"HTML(%</b></div><div class="metric"><small>Final loss</small><b>)HTML" << final.loss << R"HTML(</b></div><div class="metric"><small>Learning rate</small><b>)HTML" << final.learningRate << R"HTML(</b></div></div><section><h2>Loss</h2><canvas id="loss"></canvas></section><section><h2>Accuracy</h2><canvas id="accuracy"></canvas></section><section><h2>Learning Rate</h2><canvas id="rate"></canvas></section><script>
const series={loss:[)HTML";
    writeSeries(output, metrics, &Metric::loss);
    output << R"HTML(],accuracy:[)HTML";
    writeSeries(output, metrics, &Metric::accuracy);
    output << R"HTML(],rate:[)HTML";
    writeSeries(output, metrics, &Metric::learningRate);
    output << R"HTML(]};function draw(id,data,color){const c=document.getElementById(id),r=devicePixelRatio||1,w=c.clientWidth*r,h=c.clientHeight*r;c.width=w;c.height=h;const x=c.getContext('2d'),min=Math.min(...data),max=Math.max(...data),span=max-min||1;x.clearRect(0,0,w,h);x.strokeStyle='#dce1e8';x.lineWidth=r;for(let i=0;i<5;i++){const y=20*r+i*(h-40*r)/4;x.beginPath();x.moveTo(48*r,y);x.lineTo(w-16*r,y);x.stroke()}x.strokeStyle=color;x.lineWidth=2*r;x.beginPath();data.forEach((v,i)=>{const px=48*r+i*(w-64*r)/Math.max(1,data.length-1),py=20*r+(max-v)*(h-40*r)/span;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle='#637083';x.font=12*r+'px sans-serif';x.fillText(max.toPrecision(4),4*r,24*r);x.fillText(min.toPrecision(4),4*r,h-12*r)}Object.entries(series).forEach(([id,data])=>draw(id,data,id==='loss'?'#d85c41':id==='accuracy'?'#18836a':'#3467c8'));</script></main></body></html>)HTML";
    return output ? 0 : 1;
}
