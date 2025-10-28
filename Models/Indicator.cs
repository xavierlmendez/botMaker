namespace botMaker.strategyBacktesterLib;

public class Indicator
{
    private string ticker;

    private Trigger[] triggers;
    
    public Indicator(string ticker,  Trigger[] triggers)
    {
        this.ticker = ticker;
        this.triggers = triggers;
    }
    
    public string getTicker() => this.ticker;
}

