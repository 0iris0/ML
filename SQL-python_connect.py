import mysql.connector
connection = mysql.connector.connect(host="127.0.0.1",
                                     port="3306",
                                     user="root",
                                     password="12369",
                                     database="practice")
cursor = connection.cursor()

# 創建practice資料庫
# cursor.execute("CREATE DATABASE `practice`;")
# cursor.execute("SHOW DATABASES;")

# 將回傳資料取出
# records = cursor.fetchall()
# for r in records:
#     print(r)

# 選擇practice資料庫
# cursor.execute("USE `practice`;")

# 創建hey表格
# cursor.execute("CREATE TABLE `hey`(hey INT);")

# 選擇hey表格所有資料
# cursor.execute("SELECT*FROM `hey`;")
# info = cursor.fetchall()
# for d in info:
#     print(d)

# 新增
cursor.execute("INSERT INTO `hey` VALUES(9)")

# 修改
# cursor.execute("UPDATE `hey` SET `manger ID`=NULL WHERE `branch ID` =4;")

# 刪除
# cursor.execute("DELETE FROM `hey` WHERE `branch ID`=5;")

cursor.close()
connection.commit()  # 會動到表格內容要commit才會提交
connection.close()
