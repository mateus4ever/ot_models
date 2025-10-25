Feature: BacktestOrchestrator Methods Testing
  As a developer
  I want to test only the methods in backtest_orchestrator.py
  So that I can verify the BacktestOrchestrator logic works independently

Background:
  Given config files are available in config/backtest

Scenario: BacktestOrchestrator initialization
  When BacktestOrchestrator is initialized
  Then orchestrator should be ready

Scenario: Run multi-strategy backtest successfully
  Given data_management config points to data/stock/
  When BacktestOrchestrator is initialized
  And multi-strategy backtest is executed
  Then backtest results should be returned
  And results should contain performance metrics
