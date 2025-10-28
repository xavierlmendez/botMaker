using botMaker.strategyBacktesterLib;

namespace botMaker.services;
// migrate to portfolio project BE service
public class StrategyBacktesterService
{
    // when this is called a user object should be in the context
    public Strategy createStrategy(string strategyName)
    {
        /**
         * here we want to take in the inputs passed into function
         * construct the object in memory
         * pass the object to the ORM repo to persist it
         * then return the instance of the object to the service calling
         *
         * if the process failes we want to throw and expection
         */
        
        // get user from context
        string userName = "userName";
        

        return new Strategy(userName, strategyName);
    }

    public string[] dummyReturn()
    {
        return new[] { "strat", "test", "three" };
    }
    
}