using botMaker.services;
using Microsoft.AspNetCore.Mvc;

namespace botMaker.Controllers;

[ApiController]
[Route("api/[controller]")]
public class StrategyBacktesterController : ControllerBase
{
    
    private readonly StrategyBacktesterService _service;
    
    public StrategyBacktesterController(StrategyBacktesterService service)
    {
        _service = service;
    }
    
    [HttpGet]
    public IActionResult GetAll()
    {
        var result = _service.dummyReturn();
        return Ok(result);
    }

    [HttpGet("{id:int}")]
    public IActionResult GetById(int id) => Ok(new { id, status = "ok" });

    [HttpPost]
    public IActionResult Create([FromBody] CreateStrategyBacktestRequest req)
        => CreatedAtAction(nameof(GetById), new { id = 1 }, req);
}

public record CreateStrategyBacktestRequest(string Name, int Threshold);