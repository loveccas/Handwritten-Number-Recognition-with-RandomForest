from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
import pickle

# # 创建图形界面
fig, ax = plt.subplots(figsize=(8, 8))

# # 初始化画布 (16x16 网格)
canvas_data = np.zeros((16, 16))
im = ax.imshow(canvas_data, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1, origin='upper', extent=[0, 16, 16, 0])
ax.set_xticks([])
ax.set_yticks([])

# 鼠标绘制功能
drawing = False

def on_press(event):
    global drawing
    if event.inaxes == ax:
        drawing = True
        update_canvas(event)

def on_motion(event):
    if drawing and event.inaxes == ax:
        update_canvas(event)

def on_release(event):
    global drawing
    drawing = False

def update_canvas(event):
    # 获取鼠标点击位置对应的网格坐标
    x, y = int(event.xdata), int(event.ydata)
    # print(event.xdata,event.ydata)
    # if 0 <= x < 16 and 0 <= y < 16:
    if event.inaxes == ax:
        # 设置当前网格为1 (黑色)
        canvas_data[y, x] = 1.0
        im.set_data(canvas_data)
        fig.canvas.draw_idle()
        
# 连接鼠标事件
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

def get_write_data():
    return canvas_data.flatten()

def test():
    # 添加确认按钮
    ax_confirm = plt.axes([0.2, 0.02, 0.2, 0.06])
    ax_confirm.set_yticks([])
    ax_confirm.set_xticks([])
    btn_confirm = Button(ax_confirm, 'confirm')

    # 添加重写按钮
    ax_clear = plt.axes([0.6, 0.02, 0.2, 0.06])
    ax_clear.set_yticks([])
    ax_clear.set_xticks([])
    btn_clear = Button(ax_clear, 'Rewrite')

    # # 预测结果显示
    result_text = ax.text(8, -2, "Recognition result: ", ha='center', va='center', fontsize=12)

    with open('./hand_write_rfc.pkl','rb') as f:
        model=pickle.load(f)

    def confirm_callback(event):
            # 准备输入数据: 将16x16的画布展平为256个元素的列表
            input_data = get_write_data()   #获取按下按钮时的图像数据
            # 预测
            prediction = model.predict(input_data.reshape(1,-1))
            # 显示结果
            result_text.set_text(f"Recognition result: {prediction}")
            fig.canvas.draw_idle()

    def clear_callback(event):
        global canvas_data
        canvas_data = np.zeros((16, 16))
        im.set_data(canvas_data)
        result_text.set_text("Recognition result: ")
        fig.canvas.draw_idle()

    btn_confirm.on_clicked(confirm_callback)
    btn_clear.on_clicked(clear_callback)
    plt.show()

i=0

def train_model():
    train_feature=[]
    train_label=[]

    text_result=plt.text(8, 17, str(i), ha='center', va='center', fontsize=12)

    ax_collect = plt.axes([0.2, 0.02, 0.2, 0.06])
    ax_collect.set_yticks([])
    ax_collect.set_xticks([])
    btn_collect = Button(ax_collect, 'collect')
    
    ax_finish = plt.axes([0.6, 0.02, 0.2, 0.06])
    ax_finish.set_yticks([])
    ax_finish.set_xticks([])
    btn_finish = Button(ax_finish, 'finish&&train')

    def collect_callback(event):
        global i
        global canvas_data

        data = get_write_data()   #获取按下按钮时的图像数据
        train_feature.append(data)
        train_label.append(str(i))
        i+=1
        i=i%10
        text_result.set_text(str(i))

        canvas_data = np.zeros((16, 16))
        im.set_data(canvas_data)

        fig.canvas.draw_idle()

    def train_callback(event):
        nonlocal train_feature
        nonlocal train_label

        save_data = (train_feature, train_label)
        with open('./hand_write_data_pro.pkl','wb') as f:
            pickle.dump(save_data, f)
        train_feature=np.array(train_feature)
        train_label=np.array(train_label)
        
        model=tree.DecisionTreeClassifier(max_depth=5,min_samples_split=2)
        model.fit(train_feature,train_label)
        with open('./hand_write_model_pro.pkl','wb') as f:
            pickle.dump(model, f)
        
        print('train done')

    btn_collect.on_clicked(collect_callback)
    btn_finish.on_clicked(train_callback)
    plt.show()

# train_model()
test()
