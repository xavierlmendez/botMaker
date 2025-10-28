namespace botMaker.strategyBacktesterLib;

public class Strategy
{
    private String id;
    private String user;
    private String name;
    private Double initialCapital;
    private DateTime startDate;
    private DateTime endDate;
    private Indicator[]  indicators;
    private Trigger[]  triggers;
    
    public Strategy(String user, String strategyName)
    {
        this.user = user;
        this.name = strategyName;
    }
    
    public Strategy(String user, String strategyName, Indicator[] indicators, Trigger[]  triggers)
    {
        this.user = user;
        this.name = strategyName;
        this.indicators = indicators;
        this.triggers = triggers;
    }
}