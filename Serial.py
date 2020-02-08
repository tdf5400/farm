# -*- coding : utf-8 -*-
"""
串口部分
"""

import serial
import numpy as np
import roadCal.roadCal as rc  # 状态协议

ser = None  # 串口对象，使用全局变量方便直接调用函数操作单个串口

ACQUIRE_STA = 0        # 上位机请求状态
ACQUIRE_BOTH = 1       # 上位机请求状态+状态数据

DATA_HEADER = "aa"      # 数据头
DATA_FOOTER = "55"      # 数据尾


def getPortList():
    """
    获取串口列表
    :return:
    """
    import serial.tools.list_ports
    port_list = list(serial.tools.list_ports.comports())
    # 输出
    if len(port_list) == 0:
        # print('No Port!')
        return None
    else:
        # print('Ports')
        # for i in range(0, len(port_list)):
        #     print(port_list[i])
        return port_list


def serialInit(portx, bps, timex):
    global ser
    ser = None  # 初始化对象
    # 类型判断
    if not (isinstance(portx, str) and isinstance(bps, int)
            and isinstance(timex, int)):
        print('serialInit Error!')

    # 打开串口，并得到串口对象
    ser = serial.Serial(portx, bps, timeout=timex)
    ser.write(0x11)
    return ser


def readReg():
    """
    等待指令
    :return: 指令信息(ACQUIRE_STA / ACQUIRE_BOTH)整数型
    """
    if ser is None:
        print("[serial]Serial has not Init!")
        return None

    # 读一行数据（以\r\n划分）
    temp = bytes.hex(ser.readline())

    if len(temp) > 4:
        rData = temp[0:-4]  # 去掉末尾的\r\n
        # print(f'[serial]receive {len(rData)} byte: {rData}')

        # 确认数据头数据尾
        if rData[0:2] == DATA_HEADER and \
                rData[-2:] == DATA_FOOTER:
            reg = rData[2:-2]
            # print(f'[serial]receive reg:{reg}')
            return int(reg)     # 返回整数型的数据

    return None

def sendData(Sta, Data=None):
    """
    串口发送数据
    :param Sta: 状态标志(roadCal状态表)
    :param Data: 状态数据(int16)
    :return: 成功-1 错误-0
    """
    if Data is not None:
        # 数据加工
        Data = np.int16(Data)

        if Data > 32767:        # 范围限制
            Data = 32767
        elif Data < -32768:
            Data = -32768

        Data = Data % 65535     # 正负数转换（数据不能超过范围）

        Data = hex(Data)    # 10进制转16进制
        Data = Data[2:]     # 除去'0x'
        Data = Data[-4:]
        while len(Data) < 4:  # 变为4个字符
            Data = '0' + Data

    if Sta > 127:  # 范围限制
        Sta = 127
    elif Sta < -128:
        Sta = -128
    Sta = Sta % 255     # 正负数转换（数据不能超过范围）
    Sta = hex(Sta)      # 10进制转16进制
    Sta = Sta[2:]       # 除去'0x'
    Sta = Sta[-4:]
    while len(Sta) < 2: # 变为2个字符
        Sta = '0' + Sta

    ser.write(bytes.fromhex(DATA_HEADER))   # 首字节
    ser.write(bytes.fromhex(Sta))           # 状态
    if Data is not None:                    # 状态数据
        ser.write(bytes.fromhex(Data[0:2]))
        ser.write(bytes.fromhex(Data[2:4]))
    ser.write(bytes.fromhex(DATA_FOOTER))  # 尾字节

    # hexData = bytes.fromhex(Data)


"""
TestCode
"""
# ports = getPortList()
# for i in range(0, len(ports)):
#     print(ports[i])
# com = str(input('COM?'))
# serialInit(com,115200,5)
# serialInit("COM3", 115200, 5)
# # data = readReg()
# sendData(rc.FIT_CROSS_TRUN, 0)

