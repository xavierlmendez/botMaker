using botMaker.strategyBacktesterLib;

namespace botMaker.services;
// migrate to portfolio project BE service
public class IndicatorService
{
    public Indicator[] getIndicators(string[] tickers)
    {
        // dummy code for now
        Dictionary<string, object> triggerProps = new Dictionary<string, object>
        {
            ["Price"] = 300.0
        };
        Trigger myTrigger = new Trigger(true, "Price Greater Than", triggerProps);
        Indicator myIndicator = new Indicator("NVDA", [myTrigger]);
        return [myIndicator];
    }
}