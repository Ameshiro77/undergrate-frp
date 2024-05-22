import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    if(0):
        with open("./data/log_syn.txt",'r') as f:
            data_syn = f.readlines()
        with open("./data/log_hico.txt",'r') as f:
            data_hico = f.readlines()
    else:
        with open("./data/log_resume_syn_3.txt",'r') as f:
            data_syn = f.readlines()
        with open("./data/log_resume_hico_1.txt",'r') as f:
            data_hico = f.readlines()
    
    epoch_num = min(len(data_syn),len(data_hico))
    # syn
    test_mAP_syn = [eval(epoch)["test_mAP"] for epoch in data_syn]
    test_mAP_rare_syn = [eval(epoch)["test_mAP rare"] for epoch in data_syn]
    test_mAP_non_rare_syn = [eval(epoch)["test_mAP non-rare"] for epoch in data_syn]
    # hico
    test_mAP_hico = [eval(epoch)["test_mAP"] for epoch in data_hico[:epoch_num]]
    test_mAP_rare_hico = [eval(epoch)["test_mAP rare"] for epoch in data_hico[:epoch_num]]
    test_mAP_non_rare_hico = [eval(epoch)["test_mAP non-rare"] for epoch in data_hico[:epoch_num]]
    
    epochs = list(range(epoch_num))
    max_hico_full = max(test_mAP_hico)
    max_hico_rare = max(test_mAP_rare_hico)
    max_hico_non = max(test_mAP_non_rare_hico)
    print("hico full rare non-rare:",max_hico_full,max_hico_rare,max_hico_non)
    max_syn_full = max(test_mAP_syn)
    max_syn_rare = max(test_mAP_rare_syn)
    max_syn_non = max(test_mAP_non_rare_syn)
    print("syn full rare non-rare:",max_syn_full,max_syn_rare,max_syn_non)
    
    train_loss_syn = [eval(epoch)["train_loss"] for epoch in data_syn]
    train_loss_hico = [eval(epoch)["train_loss"] for epoch in data_hico]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    # 绘制map
    ax1.plot(epochs, test_mAP_syn, label='full', color='blue')
    ax1.plot(epochs, test_mAP_rare_syn, label='rare', color='green')
    ax1.plot(epochs, test_mAP_non_rare_syn, label='non-rare', color='red')
    ax1.plot(epochs, test_mAP_hico, label='full', color='blue',linestyle='--')
    ax1.plot(epochs, test_mAP_rare_hico, label='rare', color='green',linestyle='--')
    ax1.plot(epochs, test_mAP_non_rare_hico, label='non-rare', color='red',linestyle='--')

    # 添加图例
    ax1.legend()

    # 添加标题和标签
    ax1.set_title('mAP')
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(range(0,epoch_num))
    ax1.set_ylabel('mAP')

   # 添加图例和标题
   # loss
    #ax2.plot(epochs, train_loss_syn, label='train_loss', color='blue')
    # ax2.plot(epochs, train_loss_hico, label='train_loss', color='blue',linestyle='--')
    ax2.legend()
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    # 调整子图间距
    plt.tight_layout()

    # 显示图形
    plt.show()