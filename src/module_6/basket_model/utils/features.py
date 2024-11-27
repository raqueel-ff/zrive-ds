import numpy as np
import pandas as pd


def count_regulars_in_order(order: pd.DataFrame, user_regulars: pd.DataFrame) -> int:
<<<<<<<< HEAD:src/module_6/src/basket_model/utils/features.py
    return len(
        set(order.ordered_items).intersection(set(user_regulars.variant_id.values))
    )
========
    return len(set(order.ordered_items).intersection(set(user_regulars.variant_id.values)))
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/features.py


def count_regulars_in_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame
) -> np.ndarray:
    counts = []
    for _, order in orders.iterrows():
        user_regulars = regulars.loc[lambda x: x.user_id == order.user_id]
        counts += [count_regulars_in_order(order, user_regulars)]
    return np.array(counts)


def compute_basket_value(orders: pd.DataFrame, mean_item_price: float) -> float:
    return orders.item_count * mean_item_price

<<<<<<<< HEAD:src/module_6/src/basket_model/utils/features.py

# Fabrica features como el número de artículos regulares y valor del pedido
========
#Fabrica features como el número de artículos regulares y valor del pedido
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/features.py
def enrich_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    enriched_orders = orders.copy()
<<<<<<<< HEAD:src/module_6/src/basket_model/utils/features.py
    # Calcula el número de artículos regulares en cada pedido
    enriched_orders["regulars_count"] = count_regulars_in_orders(
        enriched_orders, regulars
    )
    # Valor medio de la cesta
========
    #Calcula el número de artículos regulares en cada pedido
    enriched_orders["regulars_count"] = count_regulars_in_orders(
        enriched_orders, regulars
    )
    #Valor medio de la cesta
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/features.py
    enriched_orders["basket_value"] = compute_basket_value(
        enriched_orders, mean_item_price
    )
    return enriched_orders


def build_prior_orders(enriched_orders: pd.DataFrame) -> pd.DataFrame:
<<<<<<<< HEAD:src/module_6/src/basket_model/utils/features.py
    # Crea DataFrame de pedidos anteriores para cada usuario
========
    #Crea DataFrame de pedidos anteriores para cada usuario
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/features.py
    prior_orders = enriched_orders.copy()
    prior_orders["user_order_seq_plus_1"] = prior_orders.user_order_seq + 1
    prior_orders["prior_basket_value"] = prior_orders["basket_value"]
    prior_orders["prior_item_count"] = prior_orders["item_count"]
    prior_orders["prior_regulars_count"] = prior_orders["regulars_count"]
    return prior_orders.loc[
        :,
        [
            "user_id",
            "user_order_seq_plus_1",
            "prior_item_count",
            "prior_regulars_count",
            "prior_basket_value",
        ],
    ]


def build_feature_frame(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    enriched_orders = enrich_orders(orders, regulars, mean_item_price)
    prior_orders = build_prior_orders(enriched_orders)
<<<<<<<< HEAD:src/module_6/src/basket_model/utils/features.py
    # Training features (características actuales y anteriores de los pedidos)
========
    #Training features (características actuales y anteriores de los pedidos)
>>>>>>>> d18b198d92ee7ad3bd70168acc3faf12f519925c:src/module_6/basket_model/utils/features.py
    return pd.merge(
        enriched_orders.loc[
            :,
            [
                "user_id",
                "created_at",
                "user_order_seq",
                "basket_value",
                "regulars_count",
            ],
        ],
        prior_orders,
        how="inner",
        left_on=("user_id", "user_order_seq"),
        right_on=("user_id", "user_order_seq_plus_1"),
    ).drop(["user_order_seq", "user_order_seq_plus_1"], axis=1)
