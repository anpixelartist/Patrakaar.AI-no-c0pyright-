import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with subplots
fig = make_subplots(
    rows=3, cols=3,
    specs=[[{}, {}, {}],
           [{'colspan': 3}, None, None],
           [{}, {}, {}]],
    vertical_spacing=0.05,
    horizontal_spacing=0.05
)

# Define node positions and styles
nodes = [
    # Row 1: Input and Models
    {"label": "Raw Articles<br>(1,035 Excel)", "x": 0, "y": 2, "color": "#FF6B6B", "type": "input"},
    {"label": "BART Summarizer<br>(1024 tokens)", "x": 1, "y": 3, "color": "#45B7D1", "type": "model"},
    {"label": "KeyBERT<br>(MMR diversity)", "x": 1, "y": 2, "color": "#45B7D1", "type": "model"},
    {"label": "BART-MNLI<br>(Zero-shot)", "x": 1, "y": 1, "color": "#45B7D1", "type": "model"},
    
    # Row 2: Processor
    {"label": "NLP Processor<br>processor.py<br>• Token truncation<br>• Lazy singletons<br>• Logging", "x": 0, "y": 1, "color": "#4EC9C4", "type": "process"},
    
    # Row 3: Output and APIs
    {"label": "Structured JSON<br>(9 fields)", "x": 0, "y": 0, "color": "#DDA0DD", "type": "output"},
    {"label": "FastAPI<br>api.py<br>• /articles<br>• /stats<br>• lifespan", "x": 1, "y": 0, "color": "#96CEB4", "type": "api"},
    {"label": "Streamlit UI<br>ui.py<br>• Filters<br>• Charts<br>• Colors", "x": 2, "y": 0, "color": "#FFEAA7", "type": "ui"},
]

# Add nodes as scatter points
for node in nodes:
    fig.add_trace(
        go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers+text",
            marker=dict(size=60, color=node["color"], line=dict(width=2, color="black")),
            text=node["label"],
            textposition="middle center",
            hoverinfo="text",
            showlegend=False
        ),
        row=1 if node["y"] > 1.5 else 2 if node["y"] > 0.5 else 3, 
        col=node["x"]+1
    )

# Add arrows (annotations)
arrows = [
    # Raw to Processor
    {"x": 0.2, "y": 1.5, "xref": "x1", "yref": "y1", 
     "ax": 0, "ay": -50},
    # Processor to models
    {"x": 0.8, "y": 1.2, "xref": "x1", "yref": "y1",
     "ax": 50, "ay": 80},
    # Processor to output
    {"x": 0.2, "y": 0.8, "xref": "x2", "yref": "y2",
     "ax": 0, "ay": 50},
    # Output to API
    {"x": 0.3, "y": 0.3, "xref": "x3", "yref": "y3",
     "ax": 80, "ay": 0},
    # Output to UI
    {"x": 0.7, "y": 0.3, "xref": "x3", "yref": "y3",
     "ax": 80, "ay": 0},
]

fig.update_layout(
    title_text="Patrakaar.AI Pipeline Architecture",
    title_font_size=20,
    showlegend=False,
    plot_bgcolor="white",
    height=800
)

# Add shape arrows
fig.add_shape(type="line", x0=0.2, y0=1.8, x1=0.2, y1=1.2, line=dict(width=2))
fig.add_shape(type="line", x0=0.5, y0=1.2, x1=0.8, y1=1.8, line=dict(width=2))
fig.add_shape(type="line", x0=0.5, y0=1.2, x1=0.8, y1=1.2, line=dict(width=2))
fig.add_shape(type="line", x0=0.5, y0=1.2, x1=0.8, y1=0.8, line=dict(width=2))

fig.write_image('D:/PATRAKAAR.AI/output/pipeline_architecture.png', width=1200, height=800)
print("Pipeline diagram generated!")
