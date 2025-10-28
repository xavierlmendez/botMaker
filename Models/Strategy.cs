using System;

namespace botMaker.strategyBacktesterLib;

public class Strategy
{
    private string id;
    private string user;
    private string name;
    private Double initialCapital;
    private DateTime startDate;
    private DateTime endDate;
    private Indicator[]  indicators;
    private Trigger[]  triggers;
    
    public Strategy(string user, string strategyName)
    {
        this.user = user;
        this.name = strategyName;
    }
    
    public Strategy(string user, string strategyName, Indicator[] indicators, Trigger[]  triggers)
    {
        this.user = user;
        this.name = strategyName;
        this.indicators = indicators;
        this.triggers = triggers;
    }
}