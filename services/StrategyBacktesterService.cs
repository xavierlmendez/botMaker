using System.Text.Json.Nodes;
using botMaker.strategyBacktesterLib;

namespace botMaker.services;
// migrate to portfolio project BE service
public class StrategyBacktesterService
{
    private JsonObject dummyReturnJSON = new JsonObject
    {
        ["strategyId"] = "STRAT-001",
        ["strategyName"] = "Moving Average Crossover",
        ["description"] = "Tests a simple moving average crossover strategy using short and long period averages.",
        ["tickers"] = new JsonArray
        {
            new JsonObject
            {
                ["symbol"] = "AAPL",
                ["displayName"] = "Apple Inc.",
                ["allocation"] = 0.4
            },
            new JsonObject
            {
                ["symbol"] = "MSFT",
                ["displayName"] = "Microsoft Corporation",
                ["allocation"] = 0.6
            }
        },
        ["timeframe"] = new JsonObject
        {
            ["start"] = "2024-01-01T00:00:00Z",
            ["end"] = "2024-12-31T00:00:00Z"
        },
        ["settings"] = new JsonObject
        {
            ["initialCapital"] = 100000,
            ["rebalanceFrequency"] = "monthly",
            ["benchmark"] = "SPY"
        },
        ["additionalProperties"] = new JsonObject
        {
            ["indicators"] = new JsonArray
            {
                new JsonObject
                {
                    ["name"] = "SMA",
                    ["parameters"] = new JsonObject
                    {
                        ["shortWindow"] = 20,
                        ["longWindow"] = 50
                    }
                },
                new JsonObject
                {
                    ["name"] = "RSI",
                    ["parameters"] = new JsonObject
                    {
                        ["period"] = 14,
                        ["overbought"] = 70,
                        ["oversold"] = 30
                    }
                }
            },
            ["riskManagement"] = new JsonObject
            {
                ["stopLoss"] = 0.05,
                ["takeProfit"] = 0.1,
                ["positionSizing"] = "fixed"
            },
            ["notes"] = "Backtest configured for 1-year period with monthly rebalancing."
        }
    };
    
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

    public JsonObject dummyReturn()
    {
        return dummyReturnJSON;
    }
    
}