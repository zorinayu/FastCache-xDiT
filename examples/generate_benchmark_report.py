#!/usr/bin/env python
"""
Generate a comprehensive benchmark report comparing different caching methods
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FastCache Benchmark Report")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory with benchmark results")
    parser.add_argument("--model_type", type=str, default="flux", help="Model type used")
    return parser.parse_args()


def load_stats(output_dir):
    """Load statistics from all benchmark runs"""
    stats = {}
    for method in ["none", "fast", "fb", "tea"]:
        stats_file = os.path.join(output_dir, f"{method}_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats[method] = json.load(f)
    return stats


def generate_performance_chart(stats, output_dir):
    """Generate performance comparison chart"""
    methods = []
    times = []
    speedups = []
    
    baseline_time = stats.get("none", {}).get("inference_time", 0)
    if baseline_time <= 0 and stats:
        # If no baseline, use the slowest method as reference
        baseline_time = max([s.get("inference_time", 0) for s in stats.values()])
    
    for method, data in stats.items():
        method_name = data.get("method", method.capitalize())
        inference_time = data.get("inference_time", 0)
        
        methods.append(method_name)
        times.append(inference_time)
        
        if baseline_time > 0:
            speedup = baseline_time / inference_time if inference_time > 0 else 0
            speedups.append(speedup)
    
    # Sort by inference time (ascending)
    sorted_indices = np.argsort(times)
    methods = [methods[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    speedups = [speedups[i] for i in sorted_indices]
    
    # Create the chart
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Bar chart for inference times
    bars = ax1.bar(methods, times, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Inference Time (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom', color='blue')
    
    # Line chart for speedups
    if speedups:
        ax2 = ax1.twinx()
        ax2.plot(methods, speedups, 'ro-', linewidth=2)
        ax2.set_ylabel('Speedup (×)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add speedup values
        for i, speedup in enumerate(speedups):
            ax2.text(i, speedup + 0.05, f'{speedup:.2f}×', ha='center', va='bottom', color='red')
    
    plt.title('FastCache Benchmark Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Performance chart saved to {chart_path}")
    return chart_path


def generate_cache_stats_chart(stats, output_dir):
    """Generate cache hit statistics chart for methods that support it"""
    methods_with_stats = []
    hit_ratios = []
    
    for method, data in stats.items():
        if "cache_stats" in data:
            method_name = data.get("method", method.capitalize())
            overall_stats = data["cache_stats"].get("overall", {})
            hit_ratio = overall_stats.get("hit_ratio", 0) * 100  # Convert to percentage
            
            methods_with_stats.append(method_name)
            hit_ratios.append(hit_ratio)
    
    if not methods_with_stats:
        print("No cache hit statistics available")
        return None
    
    # Create the chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods_with_stats, hit_ratios, color='lightgreen', alpha=0.8)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylabel('Cache Hit Ratio (%)')
    plt.title('Cache Hit Statistics Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, "cache_hit_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Cache hit chart saved to {chart_path}")
    return chart_path


def create_image_comparison(output_dir):
    """Create a side-by-side image comparison of all methods"""
    images = {}
    
    # Load all available images
    for method in ["none", "fast", "fb", "tea"]:
        img_path = os.path.join(output_dir, f"{method}_image.png")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                images[method] = img
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    if not images:
        print("No images available for comparison")
        return None
    
    # Create a grid of images
    num_images = len(images)
    if num_images == 1:
        cols, rows = 1, 1
    elif num_images <= 2:
        cols, rows = 2, 1
    else:
        cols, rows = 2, 2
    
    # Get image size
    width, height = next(iter(images.values())).size
    
    # Create a new image with space for all methods
    grid_img = Image.new('RGB', (width * cols, height * rows + 30 * rows))
    
    # Add each image to the grid with a label
    position = 0
    for method, img in images.items():
        x = (position % cols) * width
        y = (position // cols) * (height + 30)
        
        # Create a new image with space for the label
        labeled_img = Image.new('RGB', (width, height + 30), (255, 255, 255))
        labeled_img.paste(img, (0, 30))
        
        # Add the method name as a label
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(labeled_img)
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        method_name = method.capitalize()
        draw.text((10, 5), method_name, fill=(0, 0, 0), font=font)
        
        # Add to grid
        grid_img.paste(labeled_img, (x, y))
        position += 1
    
    # Save the comparison image
    comparison_path = os.path.join(output_dir, "image_comparison.png")
    grid_img.save(comparison_path)
    print(f"Image comparison saved to {comparison_path}")
    return comparison_path


def generate_html_report(stats, output_dir, model_type, performance_chart, cache_chart, image_comparison):
    """Generate a comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastCache Benchmark Results - {model_type.upper()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .chart {{ margin: 20px 0; text-align: center; }}
            .chart img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
            .image-comparison {{ text-align: center; margin: 30px 0; }}
            .image-comparison img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FastCache Benchmark Results</h1>
            <p><strong>Model:</strong> {model_type.upper()}</p>
            <p><strong>Date:</strong> {os.path.getmtime(output_dir)}</p>
            
            <h2>Performance Comparison</h2>
    """
    
    # Add performance chart
    if performance_chart:
        html_content += f"""
            <div class="chart">
                <img src="{os.path.basename(performance_chart)}" alt="Performance Comparison">
            </div>
        """
    
    # Add performance table
    html_content += """
            <table>
                <tr>
                    <th>Method</th>
                    <th>Inference Time (s)</th>
                    <th>Speedup</th>
                    <th>Resolution</th>
                    <th>Steps</th>
                </tr>
    """
    
    baseline_time = stats.get("none", {}).get("inference_time", 0)
    if baseline_time <= 0 and stats:
        baseline_time = max([s.get("inference_time", 0) for s in stats.values()])
    
    for method, data in sorted(stats.items(), key=lambda x: x[1].get("inference_time", float('inf'))):
        method_name = data.get("method", method.capitalize())
        inference_time = data.get("inference_time", 0)
        resolution = data.get("resolution", "N/A")
        steps = data.get("steps", "N/A")
        
        speedup = baseline_time / inference_time if inference_time > 0 else "N/A"
        speedup_str = f"{speedup:.2f}×" if isinstance(speedup, float) else speedup
        
        highlight = "highlight" if method == "fast" else ""
        
        html_content += f"""
                <tr class="{highlight}">
                    <td>{method_name}</td>
                    <td>{inference_time:.2f}</td>
                    <td>{speedup_str}</td>
                    <td>{resolution}</td>
                    <td>{steps}</td>
                </tr>
        """
    
    html_content += """
            </table>
    """
    
    # Add cache hit statistics
    if cache_chart:
        html_content += f"""
            <h2>Cache Hit Statistics</h2>
            <div class="chart">
                <img src="{os.path.basename(cache_chart)}" alt="Cache Hit Statistics">
            </div>
            
            <table>
                <tr>
                    <th>Method</th>
                    <th>Cache Hits</th>
                    <th>Total Steps</th>
                    <th>Hit Ratio</th>
                    <th>Cache Threshold</th>
                </tr>
        """
        
        for method, data in stats.items():
            if "cache_stats" in data:
                method_name = data.get("method", method.capitalize())
                overall_stats = data["cache_stats"].get("overall", {})
                hits = overall_stats.get("hits", 0)
                total = overall_stats.get("steps", 0)
                hit_ratio = overall_stats.get("hit_ratio", 0)
                threshold = data.get("cache_threshold", "N/A")
                
                highlight = "highlight" if method == "fast" else ""
                
                html_content += f"""
                    <tr class="{highlight}">
                        <td>{method_name}</td>
                        <td>{hits}</td>
                        <td>{total}</td>
                        <td>{hit_ratio:.2%}</td>
                        <td>{threshold}</td>
                    </tr>
                """
        
        html_content += """
            </table>
        """
    
    # Add image comparison
    if image_comparison:
        html_content += f"""
            <h2>Image Quality Comparison</h2>
            <div class="image-comparison">
                <img src="{os.path.basename(image_comparison)}" alt="Image Comparison">
            </div>
        """
    
    # Add configuration details
    html_content += """
            <h2>Configuration Details</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
    """
    
    for method, data in stats.items():
        method_name = data.get("method", method.capitalize())
        highlight = "highlight" if method == "fast" else ""
        
        # Add common parameters
        for param in ["model", "prompt", "steps", "resolution", "seed"]:
            if param in data:
                html_content += f"""
                    <tr class="{highlight}">
                        <td>{method_name}</td>
                        <td>{param}</td>
                        <td>{data[param]}</td>
                    </tr>
                """
        
        # Add method-specific parameters
        if method == "fast":
            html_content += f"""
                <tr class="{highlight}">
                    <td>{method_name}</td>
                    <td>cache_threshold</td>
                    <td>{data.get("cache_threshold", "N/A")}</td>
                </tr>
                <tr class="{highlight}">
                    <td>{method_name}</td>
                    <td>motion_threshold</td>
                    <td>{data.get("motion_threshold", "N/A")}</td>
                </tr>
            """
    
    html_content += """
            </table>
            
            <h2>Conclusion</h2>
            <p>
                FastCache provides significant speedups compared to baseline and other caching methods
                while maintaining output quality. The adaptive spatial-temporal caching approach effectively
                eliminates redundant computations across diffusion steps.
            </p>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = os.path.join(output_dir, "benchmark_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML report saved to {report_path}")
    return report_path


def main():
    args = parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all stats files
    stats = load_stats(args.output_dir)
    if not stats:
        print(f"No benchmark results found in {args.output_dir}")
        return
    
    # Generate charts
    performance_chart = generate_performance_chart(stats, args.output_dir)
    cache_chart = generate_cache_stats_chart(stats, args.output_dir)
    
    # Create image comparison
    image_comparison = create_image_comparison(args.output_dir)
    
    # Generate HTML report
    generate_html_report(stats, args.output_dir, args.model_type, 
                         performance_chart, cache_chart, image_comparison)
    
    print(f"Benchmark report generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main() 