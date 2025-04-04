"""
Visualization utilities for the AgentNet Customer Support System.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple


def create_agent_capability_radar(capabilities: List[float], specialization: str) -> go.Figure:
    """Create a radar chart of agent capabilities.
    
    Args:
        capabilities: List of capability values
        specialization: Agent specialization label
        
    Returns:
        Plotly figure object
    """
    # Create a radar chart using plotly
    categories = [f"Capability {i+1}" for i in range(len(capabilities))]
    
    fig = go.Figure()
    
    # Add the radar chart
    fig.add_trace(go.Scatterpolar(
        r=capabilities,
        theta=categories,
        fill='toself',
        name=specialization,
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=400,
        margin=dict(l=80, r=80, t=30, b=30)
    )
    
    return fig


def create_performance_chart(performance_data: Dict[str, float]) -> Optional[go.Figure]:
    """Create performance metrics chart.
    
    Args:
        performance_data: Dictionary of performance metrics
        
    Returns:
        Plotly figure object or None if no data
    """
    if not performance_data:
        return None
        
    # Create a plotly bar chart
    df = pd.DataFrame({
        'Metric': list(performance_data.keys()),
        'Value': list(performance_data.values())
    })
    
    fig = px.bar(
        df, 
        x='Metric', 
        y='Value',
        color='Value',
        color_continuous_scale='Blues',
        height=400
    )
    
    fig.update_layout(
        title='Agent Performance Metrics',
        xaxis_title='',
        yaxis_title='Score',
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=40, b=80)
    )
    
    return fig


def create_network_graph(network_state: Dict[str, Dict]) -> Optional[plt.Figure]:
    """Create a visualization of the agent network.
    
    Args:
        network_state: Dictionary with network state information
        
    Returns:
        Matplotlib figure object or None if no data
    """
    if not network_state:
        return None
        
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes for each agent
    for agent_id, info in network_state.items():
        G.add_node(
            agent_id, 
            specialization=info.get('specialization', 'Unknown'),
            performance=info.get('performance', 0.5)
        )
    
    # Add edges for recent interactions
    for agent_id, info in network_state.items():
        interactions = info.get('recent_interactions', [])
        for target_id in interactions:
            if target_id in network_state:
                G.add_edge(agent_id, target_id)
    
    # Create positions for the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create a matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Node colors based on specialization
    specializations = {data['specialization'] for _, data in G.nodes(data=True)}
    color_map = {spec: plt.cm.tab10(i) for i, spec in enumerate(specializations)}
    node_colors = [color_map[G.nodes[node]['specialization']] for node in G.nodes]
    
    # Node sizes based on performance
    node_sizes = [G.nodes[node]['performance'] * 1000 + 500 for node in G.nodes]
    
    # Draw the graph
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=12,
        font_weight='bold',
        arrows=True,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, 
                              label=spec)
                  for spec, color in color_map.items()]
    plt.legend(handles=legend_elements, title='Specializations', loc='upper left')
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()


def create_memory_usage_chart(memory_usage: Dict[str, Dict]) -> Optional[go.Figure]:
    """Create a chart of memory usage.
    
    Args:
        memory_usage: Dictionary with memory usage statistics
        
    Returns:
        Plotly figure object or None if no data
    """
    if not memory_usage:
        return None
        
    # Extract data for episodic and semantic memory
    episodic = memory_usage.get('episodic', {})
    semantic = memory_usage.get('semantic', {})
    
    # Create a dataframe
    data = {
        'Memory Type': ['Episodic', 'Semantic'],
        'Size': [
            episodic.get('size', 0),
            semantic.get('size', 0)
        ],
        'Max Size': [
            episodic.get('max_size', 1000),
            semantic.get('max_size', 2000)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate utilization
    df['Utilization'] = df['Size'] / df['Max Size'] * 100
    
    # Create a plotly bar chart
    fig = px.bar(
        df,
        x='Memory Type',
        y='Utilization',
        color='Memory Type',
        color_discrete_map={'Episodic': '#1976D2', 'Semantic': '#4CAF50'},
        labels={'Utilization': 'Utilization (%)'},
        height=300
    )
    
    fig.update_layout(
        title='Memory Utilization',
        xaxis_title='',
        yaxis_title='Utilization (%)',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    return fig


def create_historical_trends_chart(historical_data: pd.DataFrame) -> Optional[go.Figure]:
    """Create a chart showing historical trends.
    
    Args:
        historical_data: DataFrame with historical data
        
    Returns:
        Plotly figure object or None if no data
    """
    if historical_data is None or historical_data.empty:
        return None
        
    # Sample analysis: Issue distribution over time
    if 'Date of Resolution' in historical_data.columns and 'Issue Category' in historical_data.columns:
        # Convert date column to datetime
        historical_data['Date of Resolution'] = pd.to_datetime(
            historical_data['Date of Resolution'], 
            errors='coerce'
        )
        
        # Group by date and issue category
        issue_trends = historical_data.groupby([
            pd.Grouper(key='Date of Resolution', freq='M'),
            'Issue Category'
        ]).size().reset_index(name='Count')
        
        # Create a line chart
        fig = px.line(
            issue_trends, 
            x='Date of Resolution', 
            y='Count',
            color='Issue Category',
            markers=True,
            height=400
        )
        
        fig.update_layout(
            title='Issue Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Issues',
            legend_title='Issue Category',
            margin=dict(l=40, r=40, t=40, b=30)
        )
        
        return fig
    
    return None


def create_routing_visualization(routing_path: List[str]) -> Optional[go.Figure]:
    """Create a visualization of the agent routing path.
    
    Args:
        routing_path: List of agent IDs in routing order
        
    Returns:
        Plotly figure object or None if invalid path
    """
    if not routing_path or len(routing_path) < 2:
        return None
    
    # Extract agent names
    agent_names = [f"Agent {agent_id.split('_')[1]}" for agent_id in routing_path]
    
    # Create a graph for the routing path
    fig = go.Figure()
    
    # Create nodes for agents
    x_pos = list(range(len(routing_path)))
    y_pos = [0] * len(routing_path)
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=30,
            color=['#E3F2FD'] + ['#1976D2'] * (len(routing_path) - 2) + ['#4CAF50'],
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=agent_names,
        textposition="top center",
        name='Agents'
    ))
    
    # Add edges
    for i in range(len(routing_path) - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i+1]],
            y=[y_pos[i], y_pos[i+1]],
            mode='lines+markers',
            line=dict(width=2, color='grey'),
            marker=dict(size=0),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Query Routing Path",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    
    return fig


def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Plotly figure object
    """
    # Determine color based on confidence
    if confidence > 0.85:
        color = "green"
    elif confidence > 0.7:
        color = "orange"
    else:
        color = "red"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.7], 'color': "lightgray"},
                {'range': [0.7, 0.85], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.85
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    
    return fig


def create_task_distribution_chart(task_data: Dict[str, int]) -> go.Figure:
    """Create a chart showing task distribution among agents.
    
    Args:
        task_data: Dictionary mapping agent IDs to task counts
        
    Returns:
        Plotly figure object
    """
    # Sort by task count descending
    sorted_data = sorted(task_data.items(), key=lambda x: x[1], reverse=True)
    
    # Create a horizontal bar chart
    fig = go.Figure(go.Bar(
        x=[count for _, count in sorted_data],
        y=[agent_id for agent_id, _ in sorted_data],
        orientation='h',
        marker_color='#1976D2'
    ))
    
    fig.update_layout(
        title='Task Distribution Among Agents',
        xaxis_title='Number of Tasks',
        yaxis_title='Agent ID',
        height=max(300, len(task_data) * 40),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig 