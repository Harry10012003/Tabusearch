from docplex.mp.model import Model
import pandas as pd

# Dữ liệu đầu vào

data = {
    'Mã đơn hàng': ['1', '2', '3', '4', '5'],
    'Ngày sẵn sàng': ['01/01/2024'] * 5,
    'Ngày tới hạn': ['05/01/2024', '06/01/2024', '06/01/2024', '07/01/2024', '05/01/2024'],
    'Số công đoạn': [3, 2, 3, 2, 3],
    'CD-CuaL-': [10, 0, 12, 15, 8],
    'CD-Khoan-': [15, 20, 18, 0, 10],
    'A-TCNC-': [20, 25, 22, 30, 15]
}
df = pd.DataFrame(data)

# Tham số
l = len(df)  # Số đơn hàng: 5
n = 3  # Số trạm: 3 (CD-CuaL-, CD-Khoan-, A-TCNC-)
m_h = [2, 1, 2]  # Số máy tại mỗi trạm: CD-CuaL- (2), CD-Khoan- (1), A-TCNC- (2)
clusters = ['CD-CuaL-', 'CD-Khoan-', 'A-TCNC-']

# Chuyển ngày tới hạn thành giờ (giả định 12 giờ làm việc/ngày)
start_date = pd.to_datetime('01/01/2024', dayfirst=True)
d = [((pd.to_datetime(df['Ngày tới hạn'][j], dayfirst=True) - start_date).days + 1) * 12 for j in range(l)]

w = [5, 4, 3.5, 4, 4]  # Trọng số độ trễ của từng đơn hàng

# Xác định trình tự công đoạn và thời gian xử lý
order_sequence = []  # Danh sách trình tự công đoạn của mỗi đơn hàng
p = {}  # Thời gian xử lý tại mỗi trạm
x = {}  # Biến nhị phân xác định trạm nào cần xử lý
for j in range(l):
    seq = []
    p[j] = {}
    x[j] = {}
    for h in range(n):
        time = df[clusters[h]][j]
        if time > 0:
            seq.append(h)
            p[j][h] = time
            x[j][h] = 1
        else:
            x[j][h] = 0
    order_sequence.append(seq)

# Khởi tạo mô hình tối ưu hóa
mdl = Model('Production Scheduling')
L = 10000  # Giá trị Big-M để tuyến tính hóa các ràng buộc

# Biến quyết định
Ts = mdl.continuous_var_dict([(j, h, i) for j in range(l) for h in range(n) for i in range(m_h[h])], name='Ts')  # Thời gian bắt đầu
y = mdl.binary_var_dict([(j, h, i) for j in range(l) for h in range(n) for i in range(m_h[h])], name='y')  # Gán máy
v = mdl.binary_var_dict([(j, h, i, k) for j in range(l) for h in range(n) for i in range(m_h[h]) for k in range(l)], name='v')  # Thứ tự trên máy

# Biến phụ thuộc
C = mdl.continuous_var_list(l, name='C')  # Thời gian hoàn thành của đơn hàng
T = mdl.continuous_var_list(l, name='T')  # Độ trễ của đơn hàng
Te = mdl.continuous_var_dict([(j, h, i) for j in range(l) for h in range(n) for i in range(m_h[h])], name='Te')  # Thời gian kết thúc
Ts_k = mdl.continuous_var_dict([(i, h, k) for h in range(n) for i in range(m_h[h]) for k in range(l)], name='Ts_k')  # Thời gian bắt đầu theo thứ tự

# Hàm mục tiêu: Tối thiểu hóa tổng độ trễ có trọng số
mdl.minimize(mdl.sum(w[j] * T[j] for j in range(l)))

# Các ràng buộc
for j in range(l):
    # Ràng buộc (2): Độ trễ Tj = max(0, Cj - dj)
    mdl.add_constraint(T[j] >= C[j] - d[j])  # Tj >= Cj - dj
    mdl.add_constraint(T[j] >= 0)  # Tj >= 0

    # Ràng buộc (3): Thời gian hoàn thành Cj là thời gian kết thúc của công đoạn cuối cùng
    if order_sequence[j]:
        last_h = order_sequence[j][-1]
        mdl.add_constraint(C[j] == mdl.max(Te[j, last_h, i] for i in range(m_h[last_h])))

    for h in order_sequence[j]:
        # Ràng buộc (4): Mỗi công đoạn của đơn hàng chỉ được gán cho một máy
        mdl.add_constraint(mdl.sum(y[j, h, i] for i in range(m_h[h])) == 1)

        for i in range(m_h[h]):
            # Ràng buộc (1): Te[j,h,i] = Ts[j,h,i] + p[j,h] khi y[j,h,i] = 1 (dùng Big-M để tuyến tính hóa)
            mdl.add_constraint(Te[j, h, i] <= Ts[j, h, i] + p[j][h] + L * (1 - y[j, h, i]))
            mdl.add_constraint(Te[j, h, i] >= Ts[j, h, i] + p[j][h] - L * (1 - y[j, h, i]))

            # Ràng buộc (5): Chỉ gán máy nếu trạm nằm trong quy trình (y[j,h,i] <= x[j,h])
            mdl.add_constraint(y[j, h, i] <= x[j][h])

            # Ràng buộc (7): Tổng số đơn hàng được xếp thứ tự trên máy bằng với việc gán máy
            mdl.add_constraint(mdl.sum(v[j, h, i, k] for k in range(l)) == y[j, h, i])

            for k in range(l):
                # Ràng buộc (9): Mỗi thứ tự trên máy chỉ được gán cho tối đa một đơn hàng
                mdl.add_constraint(mdl.sum(v[j, h, i, k] for j in range(l)) <= 1)

                # Ràng buộc (10): Liên kết thời gian bắt đầu Ts[j,h,i] với Ts_k[i,h,k] theo thứ tự
                mdl.add_constraint(Ts[j, h, i] <= Ts_k[i, h, k] + L * (1 - v[j, h, i, k]))
                mdl.add_constraint(Ts[j, h, i] >= Ts_k[i, h, k] - L * (1 - v[j, h, i, k]))

                # Ràng buộc (8): Đảm bảo thứ tự trên máy (đơn hàng sau bắt đầu sau khi đơn hàng trước hoàn thành)
                if k < l - 1:
                    mdl.add_constraint(Ts_k[i, h, k + 1] >= Ts_k[i, h, k] + mdl.sum(p[j][h] * v[j, h, i, k] for j in range(l) if h in p[j]))

    # Ràng buộc (6): Đảm bảo trình tự công đoạn (công đoạn trước hoàn thành trước khi công đoạn sau bắt đầu)
    for k in range(len(order_sequence[j]) - 1):
        h1 = order_sequence[j][k]
        h2 = order_sequence[j][k + 1]
        mdl.add_constraint(mdl.max(Te[j, h1, i] for i in range(m_h[h1])) <= mdl.min(Ts[j, h2, i] for i in range(m_h[h2])))

# Giải mô hình
solution = mdl.solve()

# Xuất kết quả
if solution:
    print(f"Giá trị hàm mục tiêu (tổng độ trễ có trọng số): {mdl.objective_value:.2f}")
else:
    print("Không tìm thấy giải pháp.")

# export Gantt Chart
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Tạo danh sách để lưu dữ liệu Gantt Chart
gantt_data = []

# Duyệt qua từng đơn hàng và công đoạn
for j in range(l):
    for h in order_sequence[j]:
        for i in range(m_h[h]):
            if y[j, h, i].solution_value > 0.5:
                start_time = Ts[j, h, i].solution_value
                end_time = Te[j, h, i].solution_value
                machine_name = f"Trạm {clusters[h]}{i +1}"
                
                gantt_data.append({
                    'Đơn hàng': df['Mã đơn hàng'][j],
                    'Trạm máy': machine_name,
                    'Bắt đầu': start_time,
                    'Kết thúc': end_time,
                    'Thời gian': end_time - start_time
                })

# Chuyển dữ liệu thành DataFrame
df_gantt = pd.DataFrame(gantt_data)
# Định nghĩa thứ tự các trạm máy theo danh sách clusters
df_gantt['Trạm máy'] = pd.Categorical(df_gantt['Trạm máy'], 
                                      categories=[f"Trạm {cluster}{i +1 }" for cluster in clusters for i in range(m_h[clusters.index(cluster)])], 
                                      ordered=True)

# Sắp xếp DataFrame theo thứ tự trạm máy mong muốn

df_gantt = df_gantt.sort_values(by=['Trạm máy', 'Bắt đầu'])

# Vẽ biểu đồ Gantt
fig, ax = plt.subplots(figsize=(10, 5))

# Gán màu cho từng đơn hàng
order_colors = {order: plt.cm.tab10(i) for i, order in enumerate(df['Mã đơn hàng'])}

# Vẽ từng thanh trên biểu đồ Gantt
for index, row in df_gantt.iterrows():
    ax.barh(row['Trạm máy'], row['Thời gian'], left=row['Bắt đầu'], color=order_colors[row['Đơn hàng']])
    ax.text(row['Bắt đầu'] + row['Thời gian'] / 2, row['Trạm máy'], row['Đơn hàng'], 
            va='center', ha='center', color='white', fontsize=10, fontweight='bold')

# Thiết lập trục
ax.set_xlabel("Thời gian (giờ)")
ax.set_ylabel("Trạm máy")
ax.set_title("Gantt Chart - Chương trình tối ưu")
ax.grid(True, linestyle="--", alpha=0.6)

# Hiển thị biểu đồ
plt.show()
