namespace botMaker.strategyBacktesterLib.backtestConfig;

public class RiskManagement
{
    public double stopLoss { get; set; }
    public double takeProfit { get; set; }
    public string positionSizing { get; set; }
}