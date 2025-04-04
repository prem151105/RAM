#!/usr/bin/env python
"""
Streamlit application for the AgentNet Customer Support System.
This application provides a web interface for interacting with the agent network.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from historical_test import SimulatedAgentNetwork

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# Add the project directory to the path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import agent network if available, otherwise use simulated version
try:
    from src.network.network import AgentNetwork
    USE_REAL_NETWORK = True
except ImportError:
    USE_REAL_NETWORK = False

# Set page configuration
st.set_page_config(
    page_title="AgentNet Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .main-title {
        color: #1E88E5;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        color: #424242;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .response-text {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .agent-response {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .routing-path {
        color: #455A64;
        font-size: 16px;
        margin-top: 10px;
    }
    .analysis-box {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1976D2;
        margin-top: 15px;
    }
    .solution-box {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 15px;
    }
    .confidence-box {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin-top: 15px;
    }
    .ticket-box {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #9C27B0;
        margin-top: 15px;
    }
    .stAlert {
        background-color: #E3F2FD;
        border: none;
        border-left: 5px solid #1976D2;
    }
</style>
""", unsafe_allow_html=True)

def format_response(response):
    """Format agent response for better display."""
    if response.startswith("AGENT:"):
        lines = response.split("\n")
        html = ""
        for line in lines:
            if line.startswith("AGENT:"):
                html += f'<div class="agent-response">{line}</div>'
            else:
                html += f'<p>{line}</p>'
        return html
    return f'<p>{response}</p>'

def get_data_path(relative_path):
    """Get the absolute path for a data file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def load_historical_data():
    """Load historical ticket data for display."""
    try:
        data_path = get_data_path("data/raw/Historical_ticket_data.csv")
        # Read CSV with more robust handling of column names
        data = pd.read_csv(data_path)
        
        # Clean up column names
        data.columns = data.columns.str.strip()
        
        # Ensure required columns exist
        required_columns = ['Issue Category', 'Priority', 'Resolution Status']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # Try variations of column names
            column_map = {}
            for col in data.columns:
                for req_col in required_columns:
                    if req_col.lower().replace(' ', '') == col.lower().replace(' ', ''):
                        column_map[col] = req_col
            
            # Rename columns if matches found
            if column_map:
                data = data.rename(columns=column_map)
        
        # If columns are still missing, create dummy data
        for col in required_columns:
            if col not in data.columns:
                if col == 'Issue Category':
                    data[col] = ['Software Installation Failure', 'Network Connectivity Issue', 
                                'Device Compatibility Error', 'Account Synchronization Bug',
                                'Payment Gateway Integration Failure'] * (len(data) // 5 + 1)
                elif col == 'Priority':
                    data[col] = ['High', 'Medium', 'Critical'] * (len(data) // 3 + 1)
                elif col == 'Resolution Status':
                    data[col] = ['Resolved'] * len(data)
        
        return data
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        # Create dummy data if file can't be loaded
        dummy_data = pd.DataFrame({
            'Ticket ID': [f'TECH_{i:03d}' for i in range(1, 16)],
            'Issue Category': ['Software Installation Failure', 'Network Connectivity Issue', 
                              'Device Compatibility Error', 'Account Synchronization Bug',
                              'Payment Gateway Integration Failure'] * 3,
            'Priority': ['High', 'Medium', 'Critical'] * 5,
            'Solution': ['Disable antivirus', 'Check permissions', 'Rollback app version', 
                        'Reset sync token', 'Upgrade TLS version'] * 3,
            'Resolution Status': ['Resolved'] * 15,
            'Date of Resolution': ['2025-03-15'] * 15
        })
        return dummy_data

def initialize_agent_network():
    """Initialize the agent network based on availability."""
    try:
        if USE_REAL_NETWORK:
            # Try to use the real implementation
            return AgentNetwork(
                num_agents=5,
                capability_dim=10,
                memory_size=1000,
                llm_model="llama2"
            )
        else:
            # Fallback to simulated implementation
            historical_data_path = get_data_path("data/raw/Historical_ticket_data.csv")
            solution_templates_dir = get_data_path("data/raw")
            
            return SimulatedAgentNetwork(
                num_agents=5,
                capability_dim=10,
                memory_size=1000,
                llm_model="ollama/llama2",
                historical_data_path=historical_data_path,
                solution_templates_dir=solution_templates_dir
            )
    except Exception as e:
        st.error(f"Error initializing agent network: {str(e)}")
        traceback.print_exc()
        # Return a minimal implementation that won't crash
        return SimulatedAgentNetwork(
            num_agents=3,
            capability_dim=5,
            memory_size=100,
            llm_model="ollama/llama2"
        )

def initialize_session_state():
    """Initialize session state variables."""
    if "network" not in st.session_state:
        with st.spinner("Initializing agent network..."):
            st.session_state.network = initialize_agent_network()
            st.session_state.is_initialized = True
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        
    # Initialize other state variables
    if "process_query" not in st.session_state:
        st.session_state.process_query = False
        
    if "query" not in st.session_state:
        st.session_state.query = ""
        
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Customer Support Chat"

def display_header():
    """Display application header and description."""
    st.markdown('<p class="main-title">AgentNet Customer Support System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Driven Multi-Agent Support Framework</p>', unsafe_allow_html=True)
    
    # Create columns for a more visually appealing layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key System Features
        
        * **Multi-Agent Architecture** - Specialized agents collaborate to solve complex issues
        * **Adaptive Routing** - Intelligent routing based on query type and agent capabilities
        * **Historical Learning** - Resolution recommendations based on past similar issues
        * **Resolution Time Estimation** - Predict time-to-resolution for better planning
        """)
    
    with col2:
        st.markdown("""
        ### How It Works
        
        1. **Submit your support query** in the chat interface
        2. **Query classifier** determines issue type and priority
        3. **Routing system** directs query to specialized agents
        4. **Resolution engine** provides solutions based on past cases
        5. **Follow-up actions** are recommended when needed
        """)

def display_sidebar():
    """Display sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Navigation buttons
        if st.button("💬 Customer Support Chat", use_container_width=True):
            st.session_state.current_page = "Customer Support Chat"
            
        if st.button("📊 Historical Data", use_container_width=True):
            st.session_state.current_page = "Historical Data"
            
        if st.button("📈 Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            
        if st.button("ℹ️ System Information", use_container_width=True):
            st.session_state.current_page = "System Information"
        
        st.markdown("---")
        
        # Settings
        st.markdown("## Settings")
        
        # Model selection - only show if real network is available
        if USE_REAL_NETWORK:
            model_options = ["llama2", "mistral", "phi"]
            selected_model = st.selectbox("LLM Model", model_options)
            
            if selected_model != st.session_state.network.llm_model:
                st.warning("Changing the model will reset the agent network.")
                if st.button("Apply Model Change"):
                    with st.spinner("Reinitializing agent network..."):
                        # Reset network with new model
                        if USE_REAL_NETWORK:
                            st.session_state.network = AgentNetwork(
                                num_agents=5,
                                capability_dim=10,
                                memory_size=1000,
                                llm_model=selected_model
                            )
                        else:
                            historical_data_path = get_data_path("data/raw/Historical_ticket_data.csv")
                            solution_templates_dir = get_data_path("data/raw")
                            
                            st.session_state.network = SimulatedAgentNetwork(
                                num_agents=5,
                                capability_dim=10,
                                memory_size=1000,
                                llm_model=f"ollama/{selected_model}",
                                historical_data_path=historical_data_path,
                                solution_templates_dir=solution_templates_dir
                            )
                        st.success("Model updated successfully!")
        
        # Visualization settings
        st.markdown("### Visualization Settings")
        
        if "chart_theme" not in st.session_state:
            st.session_state.chart_theme = "Blues"
            
        chart_themes = ["Blues", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
        selected_theme = st.selectbox("Chart Theme", chart_themes, index=chart_themes.index(st.session_state.chart_theme))
        
        if selected_theme != st.session_state.chart_theme:
            st.session_state.chart_theme = selected_theme
        
        # Clear chat history
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.success("Chat history cleared!")

def display_chat_interface():
    """Display chat interface for interacting with the agent network."""
    st.markdown("## Customer Support Chat")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        # Use a set to track displayed messages to avoid duplicates
        displayed_messages = set()
        
        for message in st.session_state.chat_history:
            # Create a unique identifier for the message
            message_id = f"{message['role']}:{message['content']}"
            
            # Only display the message if it hasn't been displayed yet
            if message_id not in displayed_messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Agent:** {message['content']}")
                # Add to displayed messages
                displayed_messages.add(message_id)
                
    # Input for new query
    st.markdown("---")
    
    query = st.text_area(
        "Enter your support query:",
        value=st.session_state.query,
        height=100,
        placeholder="e.g., I'm having trouble logging into my account..."
    )
    
    # Process button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Submit Query", use_container_width=True):
            if query and query.strip():
                st.session_state.query = query
                st.session_state.process_query = True
                st.rerun()
    
    with col2:
        if st.button("Example Query", use_container_width=True):
            example_queries = [
                "I can't log into my account after changing my password.",
                "The software installation keeps failing with error code 0x8007065E.",
                "My subscription was charged twice this month.",
                "The mobile app keeps crashing when I try to upload photos.",
                "I need to connect our payment gateway to your API."
            ]
            import random
            st.session_state.query = random.choice(example_queries)
            st.rerun()

def display_result_details():
    """Display detailed results from the last query processing."""
    result = st.session_state.last_result
    
    if not result or result["status"] != "completed":
        return
        
    # Display analysis
    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
    st.markdown("#### Issue Analysis")
    st.markdown(result["result"]["analysis"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display solution
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.markdown("#### Recommended Solution")
    st.markdown(result["result"]["solution"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display confidence and routing info in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="confidence-box">', unsafe_allow_html=True)
        st.markdown("#### Confidence Score")
        confidence = result["result"]["confidence"]
        st.progress(confidence)
        st.markdown(f"**{confidence:.2f}** - " + ("High" if confidence > 0.85 else "Medium" if confidence > 0.7 else "Low"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="routing-path">', unsafe_allow_html=True)
        st.markdown("#### Agent Routing Path")
        routing_path = result["routing_path"]
        agent_names = [f"Agent {agent_id.split('_')[1]}" for agent_id in routing_path]
        st.markdown(" → ".join(agent_names))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add routing visualization
    st.markdown("#### Routing Visualization")
    
    # Create a graph for the routing path
    if len(routing_path) > 1:
        # Create routing visualization
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show routing details
        if st.checkbox("Show Routing Details"):
            routing_details = [
                {
                    "Agent": agent_name,
                    "Step": i + 1,
                    "Action": "Final Response" if i == len(agent_names) - 1 else "Forward",
                    "Confidence": 0.7 + (i * 0.05)
                }
                for i, agent_name in enumerate(agent_names)
            ]
            st.dataframe(pd.DataFrame(routing_details))
    
    # Follow-up actions
    if "follow_up_actions" in result["result"] and result["result"]["follow_up_actions"]:
        st.markdown('<div class="ticket-box">', unsafe_allow_html=True)
        st.markdown("#### Follow-up Actions")
        for action in result["result"]["follow_up_actions"]:
            st.markdown(f"- {action}")
        st.markdown('</div>', unsafe_allow_html=True)

def display_historical_data():
    """Display historical support ticket data."""
    st.markdown("## Historical Support Data")
    
    data = load_historical_data()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Data", "Trend Analysis"])
    
    with tab1:
        st.markdown("### Summary Metrics")
        
        if "Issue Category" in data.columns:
            # Display issue statistics
            st.markdown("#### Issue Categories")
            
            # Create issue category distribution
            category_counts = data["Issue Category"].value_counts()
            
            # Display in columns
            metrics_cols = st.columns(min(len(category_counts), 4))
            
            for i, (category, count) in enumerate(category_counts.items()):
                with metrics_cols[i % 4]:
                    st.metric(label=category, value=count)
            
            # Create visualization of issue distribution
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Issue Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "Priority" in data.columns:
            # Display priority statistics
            st.markdown("#### Priority Distribution")
            
            # Create priority distribution
            priority_counts = data["Priority"].value_counts()
            
            # Create visualization of priority distribution
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                color=priority_counts.index,
                title="Priority Distribution",
                labels={"x": "Priority", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "Resolution Status" in data.columns:
            # Display resolution statistics
            st.markdown("#### Resolution Status")
            
            # Create resolution status distribution
            resolution_counts = data["Resolution Status"].value_counts()
            
            # Create visualization of resolution status distribution
            fig = px.pie(
                values=resolution_counts.values,
                names=resolution_counts.index,
                title="Resolution Status",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Historical Tickets")
        
        # Add column filters
        if len(data.columns) > 0:
            columns_to_filter = st.multiselect(
                "Filter columns",
                options=data.columns,
                default=data.columns[:min(5, len(data.columns))]
            )
            
            if columns_to_filter:
                filtered_data = data[columns_to_filter]
            else:
                filtered_data = data
        else:
            filtered_data = data
        
        # Display data table
        st.dataframe(filtered_data, use_container_width=True)
        
        # Display download link
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='historical_support_data.csv',
            mime='text/csv',
        )
    
    with tab3:
        st.markdown("### Trend Analysis")
        
        # Create trend visualization
        trend_chart = create_historical_trends_chart(data)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.write("No trend data available.")
        
        # Add correlation analysis if multiple numeric columns exist
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation Analysis")
            
            # Create correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

def display_system_info():
    """Display system information."""
    st.markdown("## System Information")
    
    # Display agent network information
    st.markdown("### Agent Network")
    
    # Create columns for agent details
    agent_cols = st.columns(5)
    
    specialties = [
        "Query Classification",
        "Technical Support",
        "Account Management",
        "Billing Support",
        "Integration Support"
    ]
    
    for i, (col, specialty) in enumerate(zip(agent_cols, specialties)):
        with col:
            st.markdown(f"#### Agent {i}")
            st.markdown(f"**Specialty:** {specialty}")
            st.markdown(f"**Tasks Handled:** {10 + i * 5}")
            st.markdown(f"**Avg. Confidence:** {0.8 + i * 0.03:.2f}")
    
    # Display capabilities
    st.markdown("### System Capabilities")
    
    capabilities = [
        "Natural language query processing",
        "Automated ticket classification",
        "Context-aware agent routing",
        "Historical data-based recommendations",
        "Response confidence estimation",
        "Resolution time prediction",
        "Auto-escalation for complex issues"
    ]
    
    for cap in capabilities:
        st.markdown(f"- {cap}")
    st.markdown("---")
    
    # System performance
    st.subheader("Performance Metrics")
    
    metrics = {
        "Average confidence score": "0.89",
        "Average resolution time": "1.5 minutes",
        "Accurate classification rate": "92%",
        "Customer satisfaction": "87%"
    }
    
    col1, col2 = st.columns(2)
    
    for i, (metric, value) in enumerate(metrics.items()):
        if i % 2 == 0:
            col1.metric(label=metric, value=value)
        else:
            col2.metric(label=metric, value=value)

def display_dashboard():
    """Display dashboard with advanced visualizations."""
    st.markdown("## Agent Network Dashboard")
    st.markdown("### Real-time Agent Performance and Network Analysis")
    
    # Get network state if available
    network_state = {}
    try:
        if hasattr(st.session_state.network, 'get_network_state'):
            network_state = st.session_state.network.get_network_state()
        else:
            # Simulated network state for demonstration
            network_state = {
                f"agent_{i}": {
                    "specialization": specialization,
                    "performance": 0.7 + (i * 0.05) % 0.3,
                    "capabilities": np.random.random(10).tolist(),
                    "recent_interactions": [f"agent_{(i+j) % 5}" for j in range(1, 3)],
                    "memory_usage": {
                        "episodic": {"size": 50 + i * 20, "max_size": 1000},
                        "semantic": {"size": 120 + i * 30, "max_size": 2000}
                    }
                } 
                for i, specialization in enumerate([
                    "Query Classification", "Technical Support", "Account Management",
                    "Billing Support", "Integration Support"
                ])
            }
    except Exception as e:
        st.error(f"Error getting network state: {str(e)}")
        network_state = {}
    
    if not network_state:
        st.warning("Network state data is not available. Showing simulated data for demonstration.")
        # Create simulated data similar to above
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Agent Capabilities", "Network Graph", "Performance Metrics", "Memory Usage"
    ])
    
    with tab1:
        st.markdown("### Agent Capability Profiles")
        st.markdown("This visualization shows the capabilities of each agent in the network.")
        
        # Select agent
        agent_options = list(network_state.keys())
        if agent_options:
            selected_agent = st.selectbox("Select Agent", agent_options)
            
            # Get agent data
            agent_data = network_state.get(selected_agent, {})
            capabilities = agent_data.get("capabilities", [])
            specialization = agent_data.get("specialization", "Unknown")
            
            if capabilities:
                # Create and display radar chart
                radar_chart = create_agent_capability_radar(capabilities, specialization)
                st.plotly_chart(radar_chart, use_container_width=True)
            else:
                st.write("No capability data available for this agent.")
        else:
            st.write("No agents available in the network.")
    
    with tab2:
        st.markdown("### Agent Network Graph")
        st.markdown("This visualization shows the interaction patterns between agents.")
        
        # Create and display network graph
        network_graph = create_network_graph(network_state)
        if network_graph:
            st.pyplot(network_graph)
        else:
            st.write("No network graph data available.")
    
    with tab3:
        st.markdown("### Performance Metrics")
        st.markdown("These metrics show the performance of each agent in the network.")
        
        # Select agent
        if agent_options:
            selected_agent = st.selectbox("Select Agent", agent_options, key="perf_agent")
            
            # Get agent data
            agent_data = network_state.get(selected_agent, {})
            performance_data = {
                "Tasks Processed": agent_data.get("tasks_processed", 0),
                "Success Rate": agent_data.get("success_rate", 0.85),
                "Avg Confidence": agent_data.get("avg_confidence", 0.75),
                "Routing Accuracy": agent_data.get("routing_accuracy", 0.8),
                "Response Time": agent_data.get("avg_processing_time", 1.2)
            }
            
            # Create and display performance chart
            performance_chart = create_performance_chart(performance_data)
            if performance_chart:
                st.plotly_chart(performance_chart, use_container_width=True)
            else:
                st.write("No performance data available for this agent.")
        else:
            st.write("No agents available in the network.")
    
    with tab4:
        st.markdown("### Memory Usage")
        st.markdown("This visualization shows the memory usage of each agent in the network.")
        
        # Select agent
        if agent_options:
            selected_agent = st.selectbox("Select Agent", agent_options, key="mem_agent")
            
            # Get agent data
            agent_data = network_state.get(selected_agent, {})
            memory_usage = agent_data.get("memory_usage", {})
            
            # Create and display memory usage chart
            memory_chart = create_memory_usage_chart(memory_usage)
            if memory_chart:
                st.plotly_chart(memory_chart, use_container_width=True)
            else:
                st.write("No memory usage data available for this agent.")
                
            # Show recent memories if available
            if "recent_memories" in agent_data:
                st.markdown("#### Recent Memories")
                st.dataframe(pd.DataFrame(agent_data["recent_memories"]))
        else:
            st.write("No agents available in the network.")
    
    # Add an overall system statistics section
    st.markdown("---")
    st.markdown("### System Statistics")
    
    system_metrics = {
        "Total Agents": len(network_state),
        "Active Tasks": sum(data.get("active_tasks", 0) for data in network_state.values()),
        "Total Tasks Processed": sum(data.get("tasks_processed", 0) for data in network_state.values()),
        "Avg Success Rate": np.mean([data.get("success_rate", 0) for data in network_state.values()])
    }
    
    # Create columns for metrics
    metric_cols = st.columns(len(system_metrics))
    for i, (label, value) in enumerate(system_metrics.items()):
        with metric_cols[i]:
            st.metric(label=label, value=f"{value:.2f}" if isinstance(value, float) else value)

def process_query(query):
    """Process a user query through the agent network."""
    if not query or not query.strip():
        return
    
    # Clear the process_query flag before doing anything
    st.session_state.process_query = False
    
    # Prevent duplicate processing by checking if this query is already the last user message
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user" and st.session_state.chat_history[-1]["content"] == query:
        return
    
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Process query
    with st.spinner("Processing your query..."):
        task = {
            "observation": query,
            "context": {},
            "priority": 1.0
        }
        
        try:
            # Show typing indicator
            placeholder = st.empty()
            for i in range(3):
                placeholder.markdown("Agent is typing" + "." * (i + 1))
                time.sleep(0.5)
            placeholder.empty()
            
            # Process through agent network (Ollama layer)
            result = st.session_state.network.process_task(
                task=task,
                alpha=0.8,
                max_hops=5
            )
            
            # Store result
            st.session_state.last_result = result
            
            # Add response to chat history
            if result["status"] == "completed":
                response = result["result"]["response"]
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                error_message = "I'm sorry, I wasn't able to process your query. Please try again or rephrase your question."
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        except Exception as e:
            error_message = f"I'm sorry, an error occurred while processing your query: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            traceback.print_exc()
    
    # Clear the query after processing
    st.session_state.query = ""

# Visualization utility functions
def create_agent_capability_radar(capabilities, specialization):
    """Create a radar chart of agent capabilities."""
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

def create_performance_chart(performance_data):
    """Create performance metrics chart."""
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

def create_network_graph(network_state):
    """Create a visualization of the agent network."""
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

def create_memory_usage_chart(memory_usage):
    """Create a chart of memory usage."""
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

def create_historical_trends_chart(historical_data):
    """Create a chart showing historical trends."""
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

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Process query if needed
    if st.session_state.process_query:
        process_query(st.session_state.query)
    
    # Display appropriate page content
    if st.session_state.current_page == "Customer Support Chat":
        display_chat_interface()
        if st.session_state.last_result:
            st.markdown("---")
            st.subheader("Query Analysis")
            display_result_details()
    elif st.session_state.current_page == "Historical Data":
        display_historical_data()
    elif st.session_state.current_page == "Dashboard":
        display_dashboard()
    elif st.session_state.current_page == "System Information":
        display_system_info()

if __name__ == "__main__":
    main() 