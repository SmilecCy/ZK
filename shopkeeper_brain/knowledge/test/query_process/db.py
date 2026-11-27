from unittest import result

from pymongo import MongoClient

def get_collection():
    client = MongoClient("mongodb://192.168.88.160:27017")
    client.admin.command("ping")
    print("✅ MongoDB 连接成功")
    server_info = client.server_info()
    print(f"   版本: {server_info.get('version')}")
    print(f"   主机: 192.168.88.160:27017")


    return client["school"]["students"]
def add_student(coll, name: str, age: int, major: str):
    result = coll.insert_one({"name": name, "age": age, "major": major})
    print(f"[新增] {name}，ID: {result.inserted_id}")
def list_students(coll):
    # print(f"[调试] coll 类型: {type(coll)}")
    # print(f"[调试] 集合: {coll.database.name}.{coll.name}")
    # print(f"[调试] 文档总数: {coll.count_documents({})}")
    students = list(coll.find({}, {"_id": 0}))
    if not students:
        print("[查询] 暂无学生记录")
        return
    else:
        for student in students:
            print(f"  - {student['name']}，{student['age']} 岁，{student['major']}")
def update_student_age(coll, name: str, new_age: int):
    result = coll.update_one({"name": name}, {"$set": {"age": new_age}})
    if result.matched_count:
        print(f"[更新] {name} 的年龄已改为 {new_age}")
    else:
        print(f"[更新] 未找到学生 {name}")

def delete_student(coll, name: str):
    result = coll.delete_one({"name": name})
    if result.deleted_count:
        print(f"[删除] {name} 已删除")
    else:
        print(f"[删除] 未找到学生 {name}")

def main():
    coll = get_collection()
    coll.delete_many({})
    add_student(coll, "张三", 20, "计算机科学")
    add_student(coll, "李四", 22, "软件工程")
    add_student(coll, "王五", 21, "人工智能")
    list_students(coll)
    update_student_age(coll, "张三", 21)
    list_students(coll)
    delete_student(coll, "王五")
    print("\n--- 最终数据 ---")
    list_students(coll)
    #
    # client = MongoClient("mongodb://192.168.88.160:27017/")
    # # 连接库
    # db = client["mydb"]
    # # 建集合
    # collection = db["mycollection"]
    # print("连接成功！")
    # # 插入数据
    # # result = collection.insert_one({"name": "张三", "age": 20, "major": "计算机科学"})
    # # print(f"插入成功，ID: {result.inserted_id}")
    # # results = collection.insert_many([
    # #     {"name": "李四", "age": 22, "major": "软件工程"},
    # #     {"name": "王五", "age": 21, "major": "计算机科学"}
    # # ])
    # # print(f"插入 {len(results.inserted_ids)} 条记录")
    # result = collection.delete_one({"name": "王五"})
    # print(f"删除 {result.deleted_count} 条")
    # for doc in collection.find():
    #     print(doc)
    # # for doc in collection.find({"age": 22}):
    # #     print(doc["name"], doc["major"])
    # # for doc in collection.find().sort("age", -1).limit(3):
    # #     print(doc["name"], doc["age"])
    # # result = collection.update_one({"name": "张三"},  # 查询条件
    # #     {"$set": {"age": 25}}  # 更新操作
    # # )
    # # print(f"匹配 {result.matched_count} 条，修改 {result.modified_count} 条")
    # result = collection.update_many({"major": "计算机科学"},
    # {"$set": {"status": "在读"}})
    # print(f"修改 {result.modified_count} 条")

if __name__ == '__main__':

    main()
