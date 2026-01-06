import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="Gradient Calculator", layout="wide")
st.title("Topic 4: Gradient & Steepest Ascent Explorer")
st.markdown("### Interactive Tool for MAT201 Assignment 2")

# --- 2. 侧边栏：完全手动的用户输入区域 ---
st.sidebar.header("User Controls (Manual Input)")

# [关键点]：这里使用 text_input 让用户可以输入任意函数
# 老师要求的 "Interactive" 和 "Arbitrary function" 就体现在这里
function_input = st.sidebar.text_input(
    "Enter function f(x, y):", 
    value="x**2 + y**2", # 默认值，用户可以删掉自己写
    help="Support: +, -, *, /, x**2 (power), sin(x), cos(y), exp(x), etc."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Set Point $(x_0, y_0)$")

# [关键点]：这里让用户手动调节坐标点
x0 = st.sidebar.number_input("Input x coordinate:", value=1.0, step=0.1, format="%.2f")
y0 = st.sidebar.number_input("Input y coordinate:", value=1.0, step=0.1, format="%.2f")

# --- 3. 后台逻辑：处理任意数学函数 ---
try:
    # 定义符号变量
    x, y = sp.symbols('x y')
    
    # [核心技术]：将用户输入的字符串转化为数学表达式
    # 这一步保证了可以处理 "任意" 输入
    f_expr = sp.sympify(function_input)
    
    # 转换为 Python 可计算的函数 (用于画图和算数值)
    f_func = sp.lambdify((x, y), f_expr, 'numpy')

    # 自动计算偏导数 (Calculus Logic)
    fx_expr = sp.diff(f_expr, x)  # 对 x 求偏导
    fy_expr = sp.diff(f_expr, y)  # 对 y 求偏导

    # 代入具体数值计算
    z0 = float(f_func(x0, y0))
    fx_val = float(fx_expr.subs({x: x0, y: y0}))
    fy_val = float(fy_expr.subs({x: x0, y: y0}))
    
    # 计算梯度模长 (Magnitude)
    magnitude = np.sqrt(fx_val**2 + fy_val**2)

    # --- 4. 界面显示结果 ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Calculations")
        st.write("Based on your input, the system calculated:")
        
        # 显示解析式
        st.info(f"**Function:** $f(x,y) = {sp.latex(f_expr)}$")
        
        # 显示偏导结果
        st.markdown("**1. Partial Derivatives:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(fx_expr)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(fy_expr)}")
        
        # 显示梯度向量
        st.markdown(f"**2. Gradient at ({x0}, {y0}):**")
        st.latex(rf"\nabla f = \langle {fx_val:.4f}, {fy_val:.4f} \rangle")
        
        # 解释最速上升方向
        st.markdown("**3. Direction of Steepest Ascent:**")
        st.success(f"""
        To increase the function value **fastest**, move in the direction of vector **<{fx_val:.2f}, {fy_val:.2f}>**.
        
        The **Maximum Rate of Change** is **{magnitude:.4f}**.
        """)

    with col2:
        st.subheader("3D Visualization")
        
        # 动态生成画图数据
        # 范围根据用户输入的点自动调整，保证图像始终居中
        range_span = 2.0
        x_vals = np.linspace(x0 - range_span, x0 + range_span, 50)
        y_vals = np.linspace(y0 - range_span, y0 + range_span, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_func(X, Y)

        # 绘图
        fig = go.Figure()
        
        # 1. 曲面
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Function Surface'))
        
        # 2. 当前点
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode='markers', marker=dict(size=6, color='red'),
            name='Point (x0, y0)'
        ))
        
        # 3. 梯度向量箭头 (Steepest Ascent)
        # 稍微夸张一点长度以便观察
        scale_factor = 0.5
        fig.add_trace(go.Scatter3d(
            x=[x0, x0 + fx_val * scale_factor],
            y=[y0, y0 + fy_val * scale_factor],
            z=[z0, z0 + magnitude * scale_factor], 
            mode='lines+markers',
            line=dict(color='orange', width=6),
            name='Gradient Vector'
        ))

        fig.update_layout(
            title="3D View (Drag to Rotate)",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Input Error: Please check your function syntax.\n\nDetails: {e}")
    st.warning("Tips: Use '**' for power (e.g., x**2). Use 'sin(x)', 'exp(x)'.")
