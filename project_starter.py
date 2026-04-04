import pandas as pd
import numpy as np
import os
import time
import ast
import sys
from dotenv import load_dotenv
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",               "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        # Convert Row objects to dictionaries properly
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# Set up and load your env parameters and instantiate your model.
import os
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import re

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Helper function for fuzzy item name matching
def find_matching_item_name(requested_name: str) -> str:
    """
    Find the best matching item name from inventory catalog using fuzzy matching.
    
    Args:
        requested_name: The item name requested by the customer/agent
        
    Returns:
        The matching item name from the catalog, or the original name if no match found
    """
    requested_lower = requested_name.lower()
    
    # Get all inventory items
    inventory_df = pd.read_sql("SELECT item_name FROM inventory", db_engine)
    if inventory_df.empty:
        return requested_name
    
    inventory_items = inventory_df["item_name"].tolist()
    
    # Extract key words from requested name (remove common size/type prefixes)
    # Common patterns: "A4", "A3", "A5", "8.5x11", etc.
    key_words = []
    for word in requested_lower.split():
        # Skip common size indicators and articles
        if word not in ["a4", "a3", "a5", "8.5x11", "8.5", "x11", "the", "a", "an", "of", "for", "and", "or"]:
            if len(word) > 2:  # Only meaningful words
                key_words.append(word)
    
    # If no key words, try direct matching
    if not key_words:
        for item in inventory_items:
            if requested_lower in item.lower() or item.lower() in requested_lower:
                return item
        return requested_name
    
    # Score each inventory item based on keyword matches
    best_match = None
    best_score = 0
    
    for item in inventory_items:
        item_lower = item.lower()
        score = 0
        
        # Check if all key words appear in the item name
        matches = sum(1 for word in key_words if word in item_lower)
        score = matches / len(key_words) if key_words else 0
        
        # Bonus for exact substring match
        if requested_lower in item_lower or item_lower in requested_lower:
            score += 0.5
        
        # Bonus for matching the most important word (usually the paper type)
        if key_words and key_words[0] in item_lower:
            score += 0.3
        
        if score > best_score:
            best_score = score
            best_match = item
    
    # Return best match if score is reasonable (at least 30% match)
    if best_match and best_score >= 0.3:
        return best_match
    
    # Fallback: try direct substring matching
    for item in inventory_items:
        item_lower = item.lower()
        # Check if major keywords match
        if any(word in item_lower for word in key_words if len(word) > 4):
            return item
    
    return requested_name

# Tools for inventory agent
@tool
def check_inventory_status(as_of_date: str) -> str:
    """
    Check the current inventory status for all items.
    
    Args:
        as_of_date: The date to check inventory as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with inventory information for all items.
    """
    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return "No inventory available."
    
    # Get inventory details from database
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    
    result_lines = ["Current Inventory Status:"]
    for _, item in inventory_df.iterrows():
        item_name = item["item_name"]
        stock = inventory.get(item_name, 0)
        min_level = item["min_stock_level"]
        status = "LOW" if stock < min_level else "OK"
        result_lines.append(f"- {item_name}: {stock} units (min: {min_level}) [{status}]")
    
    return "\n".join(result_lines)

@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """
    Check the stock level of a specific item.
    
    Args:
        item_name: The name of the item to check.
        as_of_date: The date to check stock as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with stock information for the item.
    """
    # Use fuzzy matching to find the correct item name
    matched_name = find_matching_item_name(item_name)
    
    stock_info = get_stock_level(matched_name, as_of_date)
    if stock_info.empty or stock_info["current_stock"].iloc[0] == 0:
        if matched_name != item_name:
            return f"{item_name} (matched to '{matched_name}') is out of stock."
        return f"{item_name} is out of stock."
    
    stock = int(stock_info["current_stock"].iloc[0])
    
    # Get min stock level from inventory table
    inventory_df = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": matched_name}
    )
    
    min_level = inventory_df["min_stock_level"].iloc[0] if not inventory_df.empty else 0
    status = "LOW - Reorder needed" if stock < min_level else "OK"
    
    if matched_name != item_name:
        return f"{item_name} (matched to '{matched_name}'): {stock} units available (minimum level: {min_level}). Status: {status}"
    return f"{item_name}: {stock} units available (minimum level: {min_level}). Status: {status}"

@tool
def check_delivery_timeline(item_name: str, quantity: int, order_date: str) -> str:
    """
    Check the delivery timeline for ordering an item from the supplier.
    
    Args:
        item_name: The name of the item to order.
        quantity: The quantity to order.
        order_date: The date the order would be placed (YYYY-MM-DD format).
        
    Returns:
        A formatted string with delivery timeline information.
    """
    delivery_date = get_supplier_delivery_date(order_date, quantity)
    return f"If we order {quantity} units of {item_name} on {order_date}, the estimated delivery date from our supplier would be {delivery_date}."

@tool
def assess_reorder_needs(as_of_date: str) -> str:
    """
    Assess which items need to be reordered based on current stock levels.
    
    Args:
        as_of_date: The date to assess reorder needs as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string listing items that need reordering.
    """
    inventory = get_all_inventory(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    
    reorder_items = []
    for _, item in inventory_df.iterrows():
        item_name = item["item_name"]
        current_stock = inventory.get(item_name, 0)
        min_level = item["min_stock_level"]
        
        if current_stock < min_level:
            reorder_qty = max(min_level * 2, 200)  # Reorder at least 2x min or 200 units
            reorder_items.append({
                "item_name": item_name,
                "current_stock": current_stock,
                "min_level": min_level,
                "suggested_reorder": reorder_qty
            })
    
    if not reorder_items:
        return "All items are above minimum stock levels. No reordering needed at this time."
    
    result_lines = ["Items that need reordering:"]
    for item in reorder_items:
        result_lines.append(
            f"- {item['item_name']}: Current stock {item['current_stock']} "
            f"(below minimum {item['min_level']}). Suggested reorder: {item['suggested_reorder']} units"
        )
    
    return "\n".join(result_lines)

@tool
def place_stock_order(item_name: str, quantity: int, order_date: str) -> str:
    """
    Place a stock order to replenish inventory.
    
    Args:
        item_name: The name of the item to order.
        quantity: The quantity to order.
        order_date: The date of the order (YYYY-MM-DD format).
        
    Returns:
        A formatted string confirming the stock order.
    """
    # Use fuzzy matching to find the correct item name
    matched_name = find_matching_item_name(item_name)
    
    # Get unit price from inventory
    inventory_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": matched_name}
    )
    
    if inventory_df.empty:
        return f"Error: {item_name} (searched as '{matched_name}') is not in our inventory catalog."
    
    unit_price = float(inventory_df["unit_price"].iloc[0])
    total_cost = quantity * unit_price
    
    # Check cash availability
    cash = get_cash_balance(order_date)
    if cash < total_cost:
        return f"Cannot place order: Insufficient cash. Need ${total_cost:.2f}, but only have ${cash:.2f} available."
    
    try:
        transaction_id = create_transaction(
            item_name=matched_name,
            transaction_type="stock_orders",
            quantity=quantity,
            price=total_cost,
            date=order_date
        )
        
        # Get delivery date
        delivery_date = get_supplier_delivery_date(order_date, quantity)
        
        if matched_name != item_name:
            return (f"Stock order placed successfully! Transaction ID: {transaction_id}. "
                    f"Ordered {quantity} units of {item_name} (matched to '{matched_name}') at ${unit_price:.2f} per unit. "
                    f"Total cost: ${total_cost:.2f}. Estimated delivery: {delivery_date}")
        return (f"Stock order placed successfully! Transaction ID: {transaction_id}. "
                f"Ordered {quantity} units of {item_name} at ${unit_price:.2f} per unit. "
                f"Total cost: ${total_cost:.2f}. Estimated delivery: {delivery_date}")
    except Exception as e:
        return f"Error placing stock order: {str(e)}"

# Tools for quoting agent
@tool
def search_historical_quotes(search_terms: List[str], limit: int = 5) -> str:
    """
    Search for historical quotes that match the given search terms.
    
    Args:
        search_terms: List of terms to search for in quote history.
        limit: Maximum number of quotes to return (default: 5).
        
    Returns:
        A formatted string with historical quote information.
    """
    quotes = search_quote_history(search_terms, limit)
    
    if not quotes:
        return "No matching historical quotes found."
    
    result_lines = [f"Found {len(quotes)} historical quote(s):"]
    for i, quote in enumerate(quotes, 1):
        result_lines.append(f"\nQuote {i}:")
        result_lines.append(f"  Request: {quote.get('original_request', 'N/A')[:100]}...")
        result_lines.append(f"  Amount: ${quote.get('total_amount', 0):.2f}")
        result_lines.append(f"  Explanation: {quote.get('quote_explanation', 'N/A')[:150]}...")
        result_lines.append(f"  Order Size: {quote.get('order_size', 'N/A')}")
        result_lines.append(f"  Event Type: {quote.get('event_type', 'N/A')}")
    
    return "\n".join(result_lines)

@tool
def check_item_availability_for_quote(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Check if an item is available in sufficient quantity for a quote.
    
    Args:
        item_name: The name of the item to check.
        quantity: The quantity needed.
        as_of_date: The date to check availability as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string indicating availability.
    """
    # Use fuzzy matching to find the correct item name
    matched_name = find_matching_item_name(item_name)
    
    stock_info = get_stock_level(matched_name, as_of_date)
    
    if stock_info.empty:
        if matched_name != item_name:
            return f"{item_name} (searched as '{matched_name}') is not in our inventory."
        return f"{item_name} is not in our inventory."
    
    available_stock = int(stock_info["current_stock"].iloc[0])
    
    if available_stock >= quantity:
        if matched_name != item_name:
            return f"{item_name} (matched to '{matched_name}'): {available_stock} units available. Sufficient for order of {quantity} units."
        return f"{item_name}: {available_stock} units available. Sufficient for order of {quantity} units."
    else:
        shortfall = quantity - available_stock
        if matched_name != item_name:
            return f"{item_name} (matched to '{matched_name}'): Only {available_stock} units available. Shortfall of {shortfall} units for order of {quantity}."
        return f"{item_name}: Only {available_stock} units available. Shortfall of {shortfall} units for order of {quantity}."

# Tools for sales agent
@tool
def finalize_sale(item_name: str, quantity: int, unit_price: float, sale_date: str) -> str:
    """
    Finalize a sale by creating a transaction record.
    
    Args:
        item_name: The name of the item being sold.
        quantity: The quantity being sold.
        unit_price: The price per unit.
        sale_date: The date of the sale (YYYY-MM-DD format).
        
    Returns:
        A formatted string confirming the sale transaction.
    """
    # Use fuzzy matching to find the correct item name
    matched_name = find_matching_item_name(item_name)
    
    total_price = quantity * unit_price
    
    try:
        transaction_id = create_transaction(
            item_name=matched_name,
            transaction_type="sales",
            quantity=quantity,
            price=total_price,
            date=sale_date
        )
        if matched_name != item_name:
            return f"Sale finalized successfully! Transaction ID: {transaction_id}. Sold {quantity} units of {item_name} (matched to '{matched_name}') at ${unit_price:.2f} per unit. Total: ${total_price:.2f}"
        return f"Sale finalized successfully! Transaction ID: {transaction_id}. Sold {quantity} units of {item_name} at ${unit_price:.2f} per unit. Total: ${total_price:.2f}"
    except Exception as e:
        return f"Error finalizing sale: {str(e)}"

@tool
def get_item_unit_price(item_name: str) -> str:
    """
    Get the unit price for an item from the inventory catalog.
    
    Args:
        item_name: The name of the item to get the price for.
        
    Returns:
        A formatted string with the unit price information.
    """
    try:
        # Use fuzzy matching to find the correct item name
        matched_name = find_matching_item_name(item_name)
        
        inventory_df = pd.read_sql(
            "SELECT unit_price FROM inventory WHERE item_name = :item_name",
            db_engine,
            params={"item_name": matched_name}
        )
        
        if inventory_df.empty:
            return f"Error: {item_name} (searched as '{matched_name}') is not in our inventory catalog."
        
        unit_price = float(inventory_df["unit_price"].iloc[0])
        if matched_name != item_name:
            return f"{item_name} (matched to '{matched_name}'): Unit price is ${unit_price:.2f} per unit."
        return f"{item_name}: Unit price is ${unit_price:.2f} per unit."
    except Exception as e:
        return f"Error getting unit price for {item_name}: {str(e)}"

@tool
def check_cash_availability(as_of_date: str) -> str:
    """
    Check the current cash balance available for operations.
    
    Args:
        as_of_date: The date to check cash balance as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with cash balance information.
    """
    cash = get_cash_balance(as_of_date)
    return f"Current cash balance as of {as_of_date}: ${cash:.2f}"

@tool
def generate_financial_summary(as_of_date: str) -> str:
    """
    Generate a financial summary report.
    
    Args:
        as_of_date: The date to generate the report as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with financial summary information.
    """
    report = generate_financial_report(as_of_date)
    
    result_lines = [
        f"Financial Report as of {as_of_date}:",
        f"Cash Balance: ${report['cash_balance']:.2f}",
        f"Inventory Value: ${report['inventory_value']:.2f}",
        f"Total Assets: ${report['total_assets']:.2f}",
        "\nTop 5 Selling Products:"
    ]
    
    for product in report.get('top_selling_products', []):
        result_lines.append(
            f"  - {product.get('item_name', 'N/A')}: "
            f"{product.get('total_units', 0)} units, "
            f"${product.get('total_revenue', 0):.2f} revenue"
        )
    
    return "\n".join(result_lines)

# Tools for business advisor agent
@tool
def analyze_transaction_trends(as_of_date: str) -> str:
    """
    Analyze transaction trends to identify patterns and opportunities.
    
    Args:
        as_of_date: The date to analyze trends as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with transaction trend analysis.
    """
    report = generate_financial_report(as_of_date)
    
    # Get all transactions
    transactions = pd.read_sql(
        "SELECT * FROM transactions WHERE transaction_date <= :date AND transaction_type = 'sales'",
        db_engine,
        params={"date": as_of_date}
    )
    
    if transactions.empty:
        return "No sales transactions found for analysis."
    
    # Calculate metrics
    total_sales = len(transactions)
    total_revenue = transactions["price"].sum()
    avg_transaction_value = transactions["price"].mean()
    
    # Analyze inventory turnover
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    slow_moving = []
    fast_moving = []
    
    for _, item in inventory_df.iterrows():
        item_sales = transactions[transactions["item_name"] == item["item_name"]]
        if not item_sales.empty:
            sales_count = len(item_sales)
            if sales_count > 5:  # Fast moving
                fast_moving.append(item["item_name"])
            elif sales_count == 0:  # Slow moving
                slow_moving.append(item["item_name"])
    
    result_lines = [
        f"Transaction Analysis as of {as_of_date}:",
        f"Total Sales Transactions: {total_sales}",
        f"Total Revenue: ${total_revenue:.2f}",
        f"Average Transaction Value: ${avg_transaction_value:.2f}",
        f"\nFast-Moving Items: {', '.join(fast_moving[:5]) if fast_moving else 'None identified'}",
        f"Slow-Moving Items: {', '.join(slow_moving[:5]) if slow_moving else 'None identified'}"
    ]
    
    return "\n".join(result_lines)

@tool
def get_business_recommendations(as_of_date: str) -> str:
    """
    Generate business recommendations based on current operations.
    
    Args:
        as_of_date: The date to generate recommendations as of (YYYY-MM-DD format).
        
    Returns:
        A formatted string with business recommendations.
    """
    report = generate_financial_report(as_of_date)
    cash = report["cash_balance"]
    inventory_value = report["inventory_value"]
    
    recommendations = []
    
    # Cash management recommendations
    if cash < 10000:
        recommendations.append("⚠️ Low cash balance detected. Consider reducing inventory purchases or focusing on high-margin sales.")
    elif cash > 50000:
        recommendations.append("💡 High cash balance. Consider investing in inventory expansion or supplier relationships.")
    
    # Inventory recommendations
    if inventory_value < 5000:
        recommendations.append("📦 Low inventory value. Consider restocking popular items to avoid stockouts.")
    
    # Analyze top sellers
    top_products = report.get("top_selling_products", [])
    if top_products:
        top_item = top_products[0].get("item_name", "")
        recommendations.append(f"⭐ Top seller: {top_item}. Consider maintaining higher stock levels for this item.")
    
    # Check for items needing reorder
    inventory = get_all_inventory(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    low_stock_items = []
    for _, item in inventory_df.iterrows():
        stock = inventory.get(item["item_name"], 0)
        if stock < item["min_stock_level"]:
            low_stock_items.append(item["item_name"])
    
    if low_stock_items:
        recommendations.append(f"🔴 Items below minimum stock: {', '.join(low_stock_items[:3])}. Consider reordering soon.")
    
    if not recommendations:
        return "✅ Business operations are running smoothly. No immediate recommendations at this time."
    
    return "Business Recommendations:\n" + "\n".join(f"- {rec}" for rec in recommendations)

# Set up your agents and create an orchestration agent that will manage them.

class InventoryAgent(ToolCallingAgent):
    """Agent responsible for managing inventory, checking stock levels, and assessing reorder needs."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_inventory_status, check_item_stock, check_delivery_timeline, assess_reorder_needs, place_stock_order],
            model=model,
            name="inventory_agent",
            description="Agent responsible for inventory management. Can check stock levels, assess reorder needs, place stock orders, and check supplier delivery timelines."
        )

class QuotingAgent(ToolCallingAgent):
    """Agent responsible for generating quotes based on customer requests and historical data."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[search_historical_quotes, check_item_availability_for_quote, check_item_stock],
            model=model,
            name="quoting_agent",
            description="Agent responsible for generating quotes. Uses historical quote data and current inventory to create competitive pricing with bulk discounts."
        )

class SalesAgent(ToolCallingAgent):
    """Agent responsible for finalizing sales transactions."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[finalize_sale, check_cash_availability, check_item_stock, generate_financial_summary, get_item_unit_price],
            model=model,
            name="sales_agent",
            description="Agent responsible for finalizing sales transactions. Verifies inventory, checks cash availability, and creates transaction records. "
                       "When finalizing a sale, you MUST: 1) Use check_item_stock to verify availability, 2) Use get_item_unit_price to get the unit price, "
                       "3) Use finalize_sale with exact item_name, quantity, unit_price (from get_item_unit_price), and sale_date. "
                       "NEVER use $0.00 as unit_price - always retrieve it using get_item_unit_price first."
        )

class BusinessAdvisorAgent(ToolCallingAgent):
    """Agent responsible for analyzing business operations and providing recommendations."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[analyze_transaction_trends, get_business_recommendations, generate_financial_summary],
            model=model,
            name="business_advisor_agent",
            description="Agent responsible for analyzing transactions, identifying trends, and providing proactive business recommendations to improve efficiency and revenue."
        )

class OrchestratorAgent(ToolCallingAgent):
    """Orchestrator agent that coordinates all other agents to handle customer requests."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="orchestrator",
            description="Orchestrator agent that manages customer requests and delegates tasks to specialized agents."
        )
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)
        self.business_advisor = BusinessAdvisorAgent(model)
    
    def process_request(self, customer_request: str, request_date: str, customer_context: Dict = None, show_animation: bool = False) -> str:
        """
        Process a customer request by coordinating the appropriate agents.
        
        Args:
            customer_request: The customer's request text.
            request_date: The date of the request (YYYY-MM-DD format).
            
        Returns:
            A response to the customer.
        """
        request_lower = customer_request.lower()
        
        # Determine request type
        is_inquiry = any(term in request_lower for term in ["check", "inventory", "stock", "available", "have", "what do you have"])
        is_quote = any(term in request_lower for term in ["quote", "price", "cost", "pricing", "how much"])
        is_purchase = any(term in request_lower for term in ["buy", "purchase", "confirm", "place order", "accept", "i'll take", "i want to buy", "proceed", "yes", "go ahead"])
        
        # Extract items and quantities from request
        # This is a simplified extraction - in production, you'd use more sophisticated NLP
        items_mentioned = []
        quantities_mentioned = []
        
        # Try to extract quantities
        quantity_pattern = r'(\d+)\s*(?:sheets?|units?|reams?|rolls?|boxes?)'
        quantities = re.findall(quantity_pattern, request_lower)
        if quantities:
            quantities_mentioned = [int(q) for q in quantities]
        
        # Get inventory items from database
        inventory_df = pd.read_sql("SELECT item_name FROM inventory", db_engine)
        inventory_items = inventory_df["item_name"].tolist()
        
        # Check which items are mentioned (improved matching)
        for item in inventory_items:
            item_lower = item.lower()
            # Check for exact match or partial match
            if item_lower in request_lower or any(word in request_lower for word in item_lower.split() if len(word) > 3):
                items_mentioned.append(item)
        
        # Handle inventory inquiry
        if is_inquiry and not is_quote and not is_purchase:
            response = self.inventory_agent.run(
                f"Customer is asking about inventory. Request: {customer_request}. "
                f"Check the inventory status as of {request_date}. "
                f"Provide a helpful response about what's available."
            )
            return response
        
        # Handle quote request
        if is_quote and not is_purchase:
            # Show animation if enabled
            if show_animation:
                print_agent_activity("Orchestrator", "Analyzing request type", "working")
                time.sleep(0.3)
            
            # First check inventory
            if show_animation:
                print_agent_activity("InventoryAgent", "Checking stock availability", "working")
            inventory_check = self.inventory_agent.run(
                f"Check inventory for items mentioned in this request: {customer_request}. "
                f"Date: {request_date}. List what's available."
            )
            if show_animation:
                print_agent_complete("InventoryAgent")
            
            # Generate quote
            if show_animation:
                print_agent_activity("QuotingAgent", "Generating quote with historical data", "working")
            quote_response = self.quoting_agent.run(
                f"Generate a quote for this customer request: {customer_request}. "
                f"Request date: {request_date}. "
                f"Current inventory status: {inventory_check}. "
                f"Search historical quotes for similar requests to inform pricing. "
                f"Apply bulk discounts for large orders. "
                f"Provide a competitive quote with clear explanation of pricing and any discounts applied. "
                f"Be transparent about availability and delivery timelines. "
                f"If items are available, include specific quantities and prices."
            )
            if show_animation:
                print_agent_complete("QuotingAgent")
            
            # Check if items are actually available - if so, try to finalize sale
            items_available = not ("out of stock" in inventory_check.lower() or "not available" in inventory_check.lower() or "shortfall" in inventory_check.lower())

            if items_available:
                return f"{quote_response}\n\nWould you like to proceed with this order? Please confirm to finalize your purchase."

            return quote_response
        
        # Handle purchase/order confirmation
        if is_purchase:
            if show_animation:
                print_agent_activity("Orchestrator", "Processing purchase order", "working")
            
            # Verify inventory availability
            if show_animation:
                print_agent_activity("InventoryAgent", "Verifying stock availability", "working")
            inventory_check = self.inventory_agent.run(
                f"Verify inventory availability for this order: {customer_request}. "
                f"Date: {request_date}. Check if we have sufficient stock."
            )
            if show_animation:
                print_agent_complete("InventoryAgent")
            
            # Check if we can fulfill
            if "not available" in inventory_check.lower() or "out of stock" in inventory_check.lower():
                return f"I'm sorry, but we cannot fulfill your order at this time. {inventory_check}"
            
            # Finalize the sale
            if show_animation:
                print_agent_activity("SalesAgent", "Finalizing transaction", "working")
            sale_response = self.sales_agent.run(
                f"Finalize this sale: {customer_request}. "
                f"Sale date: {request_date}. "
                f"Inventory status: {inventory_check}. "
                f"Extract the items and quantities from the request. "
                f"For each item that is available, you MUST follow these steps: "
                f"1) Use check_item_stock to verify the item exists and has stock, "
                f"2) Use get_item_unit_price to retrieve the unit price (NEVER use $0.00), "
                f"3) Use finalize_sale with the exact item_name from inventory, quantity, unit_price (from step 2), and sale_date. "
                f"Create a separate transaction for each item. "
                f"Provide a confirmation with all transaction IDs and details including correct pricing."
            )
            if show_animation:
                print_agent_complete("SalesAgent")
            
            return sale_response
        
        # Default: treat as quote request
        inventory_check = self.inventory_agent.run(
            f"Check inventory for: {customer_request}. Date: {request_date}."
        )
        
        quote_response = self.quoting_agent.run(
            f"Generate a quote for: {customer_request}. "
            f"Request date: {request_date}. "
            f"Inventory status: {inventory_check}. "
            f"Search historical quotes and provide competitive pricing with bulk discounts."
        )
        
        return quote_response


# Terminal animation helper
def print_agent_activity(agent_name: str, activity: str, status: str = "working"):
    """
    Print animated agent activity to show request processing.
    
    Args:
        agent_name: Name of the agent
        activity: What the agent is doing
        status: Status of the activity (working, done, error)
    """
    status_symbols = {
        "working": "⚙️",
        "done": "✅",
        "error": "❌"
    }
    symbol = status_symbols.get(status, "⚙️")
    
    # Clear line and print with animation
    sys.stdout.write(f"\r{symbol} {agent_name}: {activity}")
    sys.stdout.flush()
    time.sleep(0.3)  # Small delay for animation effect

def print_agent_complete(agent_name: str, result: str = ""):
    """Print completion message for an agent."""
    print(f"\n✅ {agent_name} completed" + (f": {result[:50]}..." if result else ""))

def animate_request_processing(request_num: int, total: int, agent_steps: List[tuple]):
    """
    Animate the processing of a request through multiple agents.
    
    Args:
        request_num: Current request number
        total: Total number of requests
        agent_steps: List of (agent_name, activity) tuples
    """
    print(f"\n{'='*60}")
    print(f"Processing Request {request_num}/{total}")
    print(f"{'='*60}\n")
    
    for agent_name, activity in agent_steps:
        print_agent_activity(agent_name, activity, "working")
        time.sleep(0.5)  # Simulate processing time
        print_agent_complete(agent_name)
    
    print(f"\n{'─'*60}\n")

# Initialize the orchestrator
orchestrator = None

def initialize_orchestrator():
    """Initialize the orchestrator agent."""
    global orchestrator
    if orchestrator is None:
        orchestrator = OrchestratorAgent(model)
    return orchestrator

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    print("Initializing Multi-Agent System...")
    orchestrator = initialize_orchestrator()
    print("Multi-Agent System ready!")

    results = []
    total_requests = len(quote_requests_sample)
    
    # Get initial business advisor recommendations
    print("\n📊 Initial Business Advisor Analysis:")
    print("─" * 60)
    initial_recommendations = orchestrator.business_advisor.run(
        f"Analyze the business at the start. Date: {initial_date}. Provide initial recommendations."
    )
    print(initial_recommendations[:300] + "...")
    print("─" * 60)
    
    for idx, row in quote_requests_sample.iterrows():
        try:
            request_date = row["request_date"].strftime("%Y-%m-%d")
        except:
            print(f"⚠️ Skipping request {idx+1}: Invalid date")
            continue
        
        # Prepare customer context for negotiation
        customer_context = {
            "job": row.get("job", "customer"),
            "need_size": row.get("need_size", "medium"),
            "event": row.get("event", "event")
        }

        print(f"\n{'='*60}")
        print(f"Request {idx+1}/{total_requests}")
        print(f"{'='*60}")
        print(f"Customer: {customer_context['job']} organizing {customer_context['event']}")
        print(f"Order Size: {customer_context['need_size']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")
        print(f"{'─'*60}")

        # Process request with animation and negotiation
        try:
            response = orchestrator.process_request(
                row['request'],
                request_date,
                customer_context=customer_context,
                show_animation=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user. Saving partial results...")
            break
        except Exception as e:
            import traceback
            print(f"❌ Error processing request {idx+1}: {e}")
            traceback.print_exc()
            response = "I'm sorry, we were unable to process your request at this time. Please try again or contact us directly for assistance."

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"\n{'─'*60}")
        print("📋 FINAL RESPONSE TO CUSTOMER:")
        print(f"{'─'*60}")
        print(response)
        print(f"{'─'*60}")
        print(f"\n💰 Updated Cash: ${current_cash:.2f}")
        print(f"📦 Updated Inventory Value: ${current_inventory:.2f}")
        
        # Periodic business advisor check (every 10 requests)
        if (idx + 1) % 10 == 0:
            print(f"\n📊 Business Advisor Check (Request {idx+1}):")
            advisor_update = orchestrator.business_advisor.run(
                f"Analyze business performance so far. Date: {request_date}. "
                f"Provide recommendations based on transactions processed."
            )
            print(advisor_update[:200] + "...")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        
        # Save progress periodically (every 10 requests)
        if (idx + 1) % 10 == 0:
            pd.DataFrame(results).to_csv("test_results.csv", index=False)
            print(f"\n💾 Progress saved: {idx+1}/{total_requests} requests processed")

        # Small delay between requests to avoid rate limiting
        time.sleep(0.5)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    
    print("\n" + "="*60)
    print("="*60)
    print("FINAL FINANCIAL REPORT")
    print("="*60)
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory Value: ${final_report['inventory_value']:.2f}")
    print(f"Total Assets: ${final_report['total_assets']:.2f}")
    print("="*60)
    
    # Final business advisor recommendations
    print("\n📊 FINAL BUSINESS ADVISOR ANALYSIS:")
    print("─" * 60)
    final_advisor = orchestrator.business_advisor.run(
        f"Provide final business analysis and recommendations after processing all requests. "
        f"Date: {final_date}. Analyze overall performance and provide strategic recommendations."
    )
    print(final_advisor)
    print("─" * 60)

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print(f"\n✅ Test results saved to test_results.csv")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()