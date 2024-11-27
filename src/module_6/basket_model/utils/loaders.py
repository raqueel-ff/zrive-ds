import os
import pandas as pd


<<<<<<<< HEAD:src/module_6/src/basket_model/utils/loaders.py
STORAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "data"))
print(STORAGE)
========
STORAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/loaders.py


def load_orders() -> pd.DataFrame:
    orders = pd.read_parquet(os.path.join(STORAGE, "orders.parquet"))
    orders = orders.sort_values(by=["user_id", "created_at"])
    orders["item_count"] = orders.apply(lambda x: len(x.ordered_items), axis=1)
<<<<<<<< HEAD:src/module_6/src/basket_model/utils/loaders.py
    # user_order_seq valdrá 1 para la order mas antigua y n para la orden-n
========
    #user_order_seq valdrá 1 para la order mas antigua y n para la orden-n
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/loaders.py
    orders["user_order_seq"] = (
        orders.groupby(["user_id"])["created_at"].rank().astype(int)
    )
    return orders


def load_regulars() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(STORAGE, "regulars.parquet"))

<<<<<<<< HEAD:src/module_6/src/basket_model/utils/loaders.py

# Precio medio de todos los productos
========
#Precio medio de todos los productos
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/loaders.py
def get_mean_item_price() -> float:
    inventory = pd.read_parquet(os.path.join(STORAGE, "inventory.parquet"))
    return inventory.price.mean()
