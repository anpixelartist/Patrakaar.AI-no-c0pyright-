import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_raw = '#FF6B6B'
color_processed = '#4ECDC4'
color_model = '#45B7D1'
color_api = '#96CEB4'
color_ui = '#FFEAA7'
color_db = '#DDA0DD'

# Title
ax.text(8, 9.5, 'Patrakaar.AI News Intelligence Pipeline', 
        ha='center', fontsize=20, fontweight='bold', family='Arial')

# Raw Data
raw_box = FancyBboxPatch((0.5, 7), 2.5, 1.2, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='black', facecolor=color_raw, linewidth=2)
ax.add_patch(raw_box)
ax.text(1.75, 7.6, 'Raw Articles', ha='center', fontsize=12, fontweight='bold')
ax.text(1.75, 7.3, '(1,035 Excel rows)', ha='center', fontsize=9, style='italic')

# Processor Box
proc_box = FancyBboxPatch((3.5, 6.5), 3, 2, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='black', facecolor=color_processed, linewidth=2)
ax.add_patch(proc_box)
ax.text(5, 7.8, 'NLP Processor', ha='center', fontsize=14, fontweight='bold')
ax.text(5, 7.4, 'processor.py', ha='center', fontsize=10, family='monospace')
ax.text(5, 7.0, '- Token-based truncation', ha='center', fontsize=8)
ax.text(5, 6.7, '- Lazy model singletons', ha='center', fontsize=8)
ax.text(5, 6.4, '- Logging (no print())', ha='center', fontsize=8)

# Model boxes inside processor
y_model = 7.3
models = [
    ('BART\nSummarizer', 3.8, color_model),
    ('KeyBERT\nKeywords', 5.0, color_model),
    ('BART-MNLI\nClassifier', 6.2, color_model)
]
for text, x, color in models:
    box = FancyBboxPatch((x, y_model-0.4), 1.2, 0.8, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+0.6, y_model, text, ha='center', fontsize=8, fontweight='bold')

# Arrows from raw to processor
arrow1 = FancyArrowPatch((3.0, 7.6), (3.5, 7.6), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow1)
ax.text(3.25, 7.8, 'input', ha='center', fontsize=8, style='italic')

# Output JSON
json_box = FancyBboxPatch((3.5, 4.5), 3, 1.2, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color_db, linewidth=2)
ax.add_patch(json_box)
ax.text(5, 5.1, 'Structured Output', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 4.8, 'articles_output.json', ha='center', fontsize=9, family='monospace')
ax.text(5, 4.5, '(9 fields/article)', ha='center', fontsize=8, style='italic')

# Arrow from processor to output
arrow2 = FancyArrowPatch((5, 6.5), (5, 5.7), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow2)

# API Box
api_box = FancyBboxPatch((7.5, 7), 3, 1.5, 
                       boxstyle="round,pad=0.1", 
                       edgecolor='black', facecolor=color_api, linewidth=2)
ax.add_patch(api_box)
ax.text(9, 7.9, 'FastAPI Backend', ha='center', fontsize=13, fontweight='bold')
ax.text(9, 7.5, 'api.py', ha='center', fontsize=10, family='monospace')
ax.text(9, 7.1, '- /articles (filter)', ha='center', fontsize=8)
ax.text(9, 6.8, '- /stats (analytics)', ha='center', fontsize=8)
ax.text(9, 6.5, '- Lifespan events', ha='center', fontsize=8)

# Arrow from output to API
arrow3 = FancyArrowPatch((6.5, 5.1), (7.5, 7.5), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow3)
ax.text(6.8, 6.5, 'load JSON', ha='center', fontsize=8, style='italic', rotation=-30)

# UI Box
ui_box = FancyBboxPatch((7.5, 4.5), 3, 1.5, 
                      boxstyle="round,pad=0.1", 
                      edgecolor='black', facecolor=color_ui, linewidth=2)
ax.add_patch(ui_box)
ax.text(9, 5.4, 'Streamlit UI', ha='center', fontsize=13, fontweight='bold')
ax.text(9, 5.0, 'ui.py', ha='center', fontsize=10, family='monospace')
ax.text(9, 4.6, '- Topic filtering', ha='center', fontsize=8)
ax.text(9, 4.3, '- Confidence colors', ha='center', fontsize=8)
ax.text(9, 4.0, '- Plotly charts', ha='center', fontsize=8)

# Arrow from output to UI
arrow4 = FancyArrowPatch((6.5, 5.1), (7.5, 5.25), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow4)

# User boxes
ax.text(12.5, 7.9, 'Developers', ha='center', fontsize=11, fontweight='bold')
ax.text(12.5, 7.5, 'API Docs', ha='center', fontsize=9)
user1 = patches.Circle((12.5, 7.0), 0.3, color='lightgray', ec='black')
ax.add_patch(user1)

ax.text(12.5, 5.4, 'Editors', ha='center', fontsize=11, fontweight='bold')
ax.text(12.5, 5.0, 'Dashboard', ha='center', fontsize=9)
user2 = patches.Circle((12.5, 4.5), 0.3, color='lightgray', ec='black')
ax.add_patch(user2)

# Arrows to users
arrow5 = FancyArrowPatch((10.5, 7.75), (12.2, 7.75), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow5)

arrow6 = FancyArrowPatch((10.5, 5.25), (12.2, 5.25), 
                        arrowstyle='->', linewidth=2, color='black')
ax.add_patch(arrow6)

# Challenges solved box
challenge_box = FancyBboxPatch((0.5, 1), 14, 2.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='red', facecolor='#FFE5E5', linewidth=2, linestyle='--')
ax.add_patch(challenge_box)
ax.text(7.5, 3.2, 'Challenges Solved (The "Smart" Stuff)', 
        ha='center', fontsize=14, fontweight='bold', color='darkred')

challenges = [
    'Token-based truncation (fixed CUDA crashes)',
    'Content fallback (Title to Content)',
    'Lazy singletons (6GB VRAM management)',
    'Pipe-separated tags (CSV compliance)',
    'Idempotency check (skip reprocessing)',
    'All 9 output fields validated'
]
y_start = 2.8
for i, text in enumerate(challenges):
    x_pos = 1.0 if i < 2 else 5.5 if i < 4 else 10.0
    y_pos = y_start - (i % 2) * 0.4
    ax.text(x_pos, y_pos, text, fontsize=9, va='center')

# Footer
ax.text(8, 0.3, 'Built with: Python 3.12 | PyTorch 2.3.1 | Transformers 4.47.0 | KeyBERT 0.8.5', 
        ha='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig('D:/PATRAKAAR.AI/output/pipeline_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Pipeline architecture diagram generated!")
