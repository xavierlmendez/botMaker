namespace botMaker.strategyBacktesterLib;
public enum TriggerStatus
{
    Buy,
    Hold,
    Sell
}

public enum StatusCode : byte
{
    Buy = 150,
    Hold = 100,
    Sell = 50
}
public class Trigger
{
    private bool isBuyTrigger;
    private string condition;
    
    public Dictionary<string, object> triggerProperties;

    public Trigger(bool isBuyTrigger, string condition, Dictionary<string, object> triggerProperties)
    {
        this.isBuyTrigger = isBuyTrigger;
        this.condition = condition;
        this.triggerProperties = triggerProperties;
    }

    public Enum GetStatus()
    {
        if (ConditionMet())
        {
            return this.isBuyTrigger ?  StatusCode.Buy : StatusCode.Sell;
        }

        return StatusCode.Hold;
    }

    private bool ConditionMet()
    {
        /**
         * plan atm is to have this use a condition parser class allowing future expansion
         *
         * one option is to keep the trigger properties to pass relevent data
         * then use the condition parser to correctly pass the properties to the right logic
         *
         * option two is to keep this simple stupid and add a ticker
         * then the condition would just be an enum like PRICE or RMSE
         * this would make the arch simpler but loss some flexablitiy
         * and require development for new conditions 
         */
        
        
        return false;
    }
}