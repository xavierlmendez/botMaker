using Microsoft.AspNetCore.Mvc;

namespace botMaker.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AlertsController : ControllerBase
{
    [HttpGet]                       // GET /api/alerts
    public IActionResult GetAll() => Ok(new[] { "ping", "pong" });

    [HttpGet("{id:int}")]           // GET /api/alerts/123
    public IActionResult GetById(int id) => Ok(new { id, status = "ok" });

    [HttpPost]                      // POST /api/alerts
    public IActionResult Create([FromBody] CreateAlertRequest req)
        => CreatedAtAction(nameof(GetById), new { id = 1 }, req);
}

public record CreateAlertRequest(string Name, int Threshold);