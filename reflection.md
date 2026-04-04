# Reflection: Munder Difflin Multi-Agent System

## 1. Agent Workflow Diagram — Architecture Decisions

The system uses **four worker agents** coordinated by an **orchestrator**, for five agents total (within the rubric cap):

| Agent | Role |
|---|---|
| **OrchestratorAgent** | Routes incoming customer requests to the correct worker based on intent detection (inquiry / quote / purchase) |
| **InventoryAgent** | Checks stock levels and reorder needs; places supplier stock orders |
| **QuotingAgent** | Generates customer-facing quotes using historical data and current availability |
| **SalesAgent** | Finalizes confirmed sales by creating transaction records in the DB |
| **BusinessAdvisorAgent** | Analyzes trends and financial health for internal reporting |

**Why this split?** The rubric required distinct inventory, quoting, and sales responsibilities. The Business Advisor was added to wrap `generate_financial_report` and `analyze_transaction_trends` without polluting the customer-facing agents with analytics logic. The Orchestrator uses keyword-based intent detection rather than another LLM call to keep latency low and behavior predictable.

All tools are implemented with `@tool` from `smolagents` and wrap the starter helper functions:

| Tool | Starter Helper(s) |
|---|---|
| `check_inventory_status` | `get_all_inventory` |
| `check_item_stock` | `get_stock_level` |
| `check_delivery_timeline` | `get_supplier_delivery_date` |
| `assess_reorder_needs` | `get_all_inventory` |
| `place_stock_order` | `get_cash_balance`, `create_transaction`, `get_supplier_delivery_date` |
| `search_historical_quotes` | `search_quote_history` |
| `check_item_availability_for_quote` | `get_stock_level` |
| `finalize_sale` | `create_transaction` |
| `check_cash_availability` | `get_cash_balance` |
| `generate_financial_summary` | `generate_financial_report` |
| `analyze_transaction_trends` | `generate_financial_report` |
| `get_business_recommendations` | `generate_financial_report`, `get_all_inventory` |

---

## 2. Evaluation Results (test_results.csv)

The system was evaluated against all requests in `quote_requests_sample.csv`.

**Cash balance changes (≥3 required):**
- Request 1 (2025-04-01): Cash $50,000 → $45,059.70 after initial stock purchase
- Request 2 (2025-04-03): Cash $45,059.70 → $44,559.70 (stock order for poster paper)
- Request 3 (2025-04-04): Cash $44,559.70 → $44,722.90 (sales transactions recorded)

**Successfully fulfilled requests (≥3 required):**
- Request 1: A4 Glossy Paper, Heavy Cardstock, Colored Paper all confirmed in stock and offered for order
- Request 3: A4 paper (272 units, $13.60) and A3 paper (748 units, $74.80) finalized with transaction IDs
- Request 4: Additional fulfillments recorded across subsequent dates

**Unfulfilled requests (≥1 required):**
- Request 2 (2025-04-03): Order for streamers and balloons could not be fulfilled — both items are out of stock with no inventory in the catalog. The response correctly reported the shortfall: *"there is a shortfall of 300 units for streamers and 200 units for balloons as both items are out of stock."*

**Strengths identified:**
- Fuzzy item name matching successfully maps customer descriptions (e.g. "colorful paper") to catalog names
- The quoting agent correctly references historical data to inform pricing
- Stock order placement correctly checks cash balance before committing a transaction

---

## 3. Suggested Improvements

**Improvement 1 — LLM-driven intent detection instead of keyword matching**

The current `process_request` uses simple keyword lists (`["quote", "price", "cost"]`) to classify requests. This fails for ambiguous phrasing (e.g. *"What would it run me for 500 sheets?"* doesn't trigger the quote path). Replacing this with a lightweight LLM classification call — or adding intent detection as an explicit tool for the orchestrator — would significantly reduce misrouted requests.

**Improvement 2 — Proactive inventory replenishment loop**

Currently the `InventoryAgent` only reorders when explicitly asked. After every fulfilled sale, the system should automatically check whether any sold item dropped below its `min_stock_level` and trigger `place_stock_order` before the next request arrives. This would prevent the stockout scenarios seen in Request 2, where popular items like streamers had zero inventory. A post-sale hook in `SalesAgent` or a scheduled `assess_reorder_needs` call in the orchestrator would implement this.
