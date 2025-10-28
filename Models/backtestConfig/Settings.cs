using System;
using System.Collections.Generic;

namespace botMaker.strategyBacktesterLib.backtestConfig;

public class Settings
{
    public int initialCapital { get; set; }
    public string rebalanceFrequency { get; set; }
    public string benchmark { get; set; }
}

// Root myDeserializedClass = JsonConvert.DeserializeObject<Root>(myJsonResponse);
public class AdditionalProperties
{
    public List<Indicator> indicators { get; set; }
    public RiskManagement riskManagement { get; set; }
    public string notes { get; set; }
}

public class Indicator
{
    public string name { get; set; }
    public Parameters parameters { get; set; }
}

public class Parameters
{
    public int shortWindow { get; set; }
    public int longWindow { get; set; }
    public int? period { get; set; }
    public int? overbought { get; set; }
    public int? oversold { get; set; }
}

public class Root
{
    public string strategyId { get; set; }
    public string strategyName { get; set; }
    public string description { get; set; }
    public List<Ticker> tickers { get; set; }
    public Timeframe timeframe { get; set; }
    public Settings settings { get; set; }
    public AdditionalProperties additionalProperties { get; set; }
}

public class Ticker
{
    public string symbol { get; set; }
    public string displayName { get; set; }
    public double allocation { get; set; }
}

public class Timeframe
{
    public DateTime start { get; set; }
    public DateTime end { get; set; }
}

