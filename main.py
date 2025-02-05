from fastapi import FastAPI, HTTPException
import requests
from typing import List

app = FastAPI()

# Sample product data
products = [
    {"id": 1, "name": "Laptop", "price": 1000},
    {"id": 2, "name": "Smartphone", "price": 500},
    {"id": 3, "name": "Tablet", "price": 300},
]

# Sample user data
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]

# Simulated database for orders
orders = []

# Product Service
@app.get("/products", response_model=List[dict])
def get_products():
    """Retrieve a list of products."""
    return {"products": products}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Retrieve details of a specific product."""
    product = next((p for p in products if p["id"] == product_id), None)
    if product:
        return product
    raise HTTPException(status_code=404, detail="Product not found")

# Order Service
@app.post("/orders")
def create_order(product_id: int, quantity: int):
    """Create an order for a product."""
    response = requests.get(f"http://localhost:8001/products/{product_id}")
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Product not found")

    product = response.json()
    total_price = product["price"] * quantity
    order = {"product_id": product_id, "quantity": quantity, "total_price": total_price}
    orders.append(order)
    return {"message": "Order created successfully", "order": order}

@app.get("/orders")
def get_orders():
    """Retrieve all orders."""
    return {"orders": orders}

# User Service
@app.get("/users", response_model=List[dict])
def get_users():
    """Retrieve a list of users."""
    return {"users": users}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    """Retrieve details of a specific user."""
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        return user
    raise HTTPException(status_code=404, detail="User not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)