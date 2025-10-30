import textwrap

from pathlib import Path

from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import SqlTask
from datus.tools.semantic_models import SemanticModelRepository


def _make_agent_config() -> AgentConfig:
    return AgentConfig(
        nodes={},
        target="default",
        models={
            "default": {
                "type": "openai",
                "base_url": "https://example.com",
                "api_key": "dummy",
                "model": "dummy",
            }
        },
        namespace={},
        semantic_models={
            "default_strategy": "auto",
            "default_volume": "",
            "default_directory": "",
            "allow_local_path": True,
            "prompt_max_length": 12000,
        },
        storage={},
    )


def test_semantic_model_repository_loads_local_file(tmp_path):
    agent_config = _make_agent_config()
    semantic_content = textwrap.dedent(
        """
        semantic_models:
          - name: revenue_model
            description: Revenue metrics by customer.
            model: ref('fct_revenue')
            dimensions:
              - name: customer_id
                description: Customer identifier
            measures:
              - name: total_revenue
                description: Sum of all revenue
        metrics:
          - name: total_revenue_metric
            description: Total revenue metric
        """
    ).strip()

    semantic_file = Path(tmp_path) / "revenue.yaml"
    semantic_file.write_text(semantic_content, encoding="utf-8")

    sql_task = SqlTask(
        id="test-task",
        database_type="duckdb",
        task="List top customers by revenue",
        database_name="duck",
        context_strategy="semantic_model",
        semantic_model_local_path=str(semantic_file),
    )

    repository = SemanticModelRepository(agent_config)
    payload = repository.load(sql_task)

    assert payload is not None
    assert payload.prompt_text  # Ensure prompt text populated
    assert "total_revenue" in payload.prompt_text
    assert any("total_revenue" in measure for measure in payload.measures)
    assert payload.source.endswith("revenue.yaml")


def test_semantic_model_repository_skips_when_no_location():
    agent_config = _make_agent_config()
    sql_task = SqlTask(
        id="task-2",
        database_type="duckdb",
        task="Show sales by day",
        database_name="duck",
        context_strategy="auto",
    )

    repository = SemanticModelRepository(agent_config)
    payload = repository.load(sql_task)
    assert payload is None


def test_semantic_model_repository_list_local(tmp_path):
    agent_config = _make_agent_config()
    repo = SemanticModelRepository(agent_config)

    (tmp_path / "finance.yaml").write_text("semantic_models: []", encoding="utf-8")
    (tmp_path / "sales.yml").write_text("semantic_models: []", encoding="utf-8")

    files = repo.list_models(local_dir=str(tmp_path))
    assert files == ["finance.yaml", "sales.yml"]


def test_semantic_model_repository_parses_analyst_spec(tmp_path):
    agent_config = _make_agent_config()
    repo = SemanticModelRepository(agent_config)

    spec_content = """
name: Retail Analytics
description: Demo spec
comments: External facing model
tables:
  - name: fact_sales
    description: Sales facts
    base_table:
      database: analytics
      schema: mart
      table: fact_sales
    dimensions:
      - name: product_id
        description: Product identifier
        data_type: INT
        synonyms: [pid]
      - name: customer_id
        description: Customer identifier
        data_type: INT
    time_dimensions:
      - name: order_date
        description: Order date
        data_type: DATE
    facts:
      - name: total_amount
        description: Total order amount
        expr: SUM(amount)
        data_type: DECIMAL
    metrics:
      - name: total_income
        description: Income per table
        expr: SUM(total_amount)
        access_modifier: public_access
    filters:
      - name: current_year
        description: Current year orders
        expr: DATE_PART('year', order_date) = DATE_PART('year', CURRENT_DATE)
  - name: dim_product
    description: Product dimension
    base_table:
      database: analytics
      schema: mart
      table: dim_product
    dimensions:
      - name: product_id
        description: Product identifier
        data_type: INT
      - name: category
        description: Product category
        data_type: STRING
relationships:
  - name: product_lookup
    left_table: fact_sales
    right_table: dim_product
    relationship_columns:
      - left_column: product_id
        right_column: product_id
    join_type: inner
    relationship_type: many_to_one
metrics:
  - name: total_income
    description: Total revenue across model
    expr: SUM(total_amount)
verified_queries:
  - name: tablet_total_income
    question: 平板的 total_income
    sql: SELECT SUM(total_amount) FROM analytics.mart.fact_sales WHERE category = '平板';
"""
    spec_path = tmp_path / "retail.yaml"
    spec_path.write_text(spec_content, encoding="utf-8")

    sql_task = SqlTask(
        id="spec",
        database_type="duckdb",
        task="Load spec",
        database_name="duck",
        context_strategy="semantic_model",
        semantic_model_local_path=str(spec_path),
    )

    payload = repo.load(sql_task)
    assert payload is not None
    assert payload.name == "Retail Analytics"
    assert payload.logical_tables
    fact_table = next((t for t in payload.logical_tables if t.name == "fact_sales"), None)
    assert fact_table is not None
    assert fact_table.base_table is not None
    assert fact_table.base_table.table == "fact_sales"
    assert any(dim.name == "product_id" for dim in fact_table.dimensions)
    assert payload.relationships
    assert payload.model_metrics
    assert payload.verified_queries
    assert "Relationships:" in payload.prompt_text
    assert "Verified Queries:" in payload.prompt_text
