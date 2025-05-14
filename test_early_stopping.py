import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入EarlyStopping类
from examples.integrate_advanced_transgnn import EarlyStopping

# 如果成功导入，打印成功信息
print("成功导入EarlyStopping类！")
print("类定义如下:")
print(f"patience参数默认值: {EarlyStopping.__init__.__defaults__[0]}")
print(f"verbose参数默认值: {EarlyStopping.__init__.__defaults__[1]}")
print(f"dataset_name参数默认值: {EarlyStopping.__init__.__defaults__[2]}")
print(f"delta参数默认值: {EarlyStopping.__init__.__defaults__[3]}")

print("\nEarlyStopping类已成功添加到integrate_advanced_transgnn.py文件中！")