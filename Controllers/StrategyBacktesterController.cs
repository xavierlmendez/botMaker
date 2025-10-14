using Microsoft.AspNetCore.Mvc;

namespace botMaker.Controllers;

[ApiController]
[Route("api/[controller]")]
public class StrategyBacktesterController : ControllerBase
{
    [HttpGet]
    public IActionResult GetAll() => Ok(new[] { "strat", "test" });

    [HttpGet("{id:int}")]
    public IActionResult GetById(int id) => Ok(new { id, status = "ok" });

    [HttpPost]
    public IActionResult Create([FromBody] CreateStrategyBacktestRequest req)
        => CreatedAtAction(nameof(GetById), new { id = 1 }, req);
}

public record CreateStrategyBacktestRequest(string Name, int Threshold);