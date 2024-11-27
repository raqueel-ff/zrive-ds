import pytest
from src.module_6.src.data_model import Request
import os
import sys
from httpx import AsyncClient

# To put the directory of the code to test
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src/module_6"))
)

USER_ID = "1296e1e72f7f43ff28d7d285f880ad4d213fa8139233c78ad2673af4cc0c1c297180254da494cc3b934858347822be437eeb072acb1403e9476dc0b2f6a83cbe"


# Testing /status handle
@pytest.mark.asyncio
async def test_status():
    async with AsyncClient() as client:
        response = await client.request(
            method="GET", timeout=20, url="http://localhost:8000/status/"
        )
    assert response.status_code == 200


# Testing correct USER_ID in /predict handle
@pytest.mark.asyncio
async def test_succesful_response():
    # input_request contiene el USER_ID en el formato que lo debe recibir la API
    input_request = Request(user_id=USER_ID).model_dump()

    async with AsyncClient() as client:
        response = await client.request(
            method="POST",
            timeout=20,
            url="http://localhost:8000/predict/",
            json=input_request,
        )
    assert response.status_code == 200


# Testing incorrect USER_ID in /predict handle
@pytest.mark.asyncio
async def test_not_existet_user():
    input_request = Request(user_id="a").model_dump()

    async with AsyncClient() as client:
        response = await client.request(
            method="POST",
            timeout=20,
            url="http://localhost:8000/predict/",
            json=input_request,
        )
    assert response.status_code == 500
