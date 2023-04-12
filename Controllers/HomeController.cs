using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using onnx.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace onnx.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly OnnxModelScorer _scorer;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;

            // Load the ONNX model
            var modelPath = "C:/Users/emily/source/repos/onnx/supervised.onnx";
            _scorer = new OnnxModelScorer(modelPath);
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        [HttpPost]
        public IActionResult Predict(IFormCollection form)
        {
            // Get the form data
            float[] inputValues = new float[15];
            for (int i = 0; i < 15; i++)
            {
                if (!float.TryParse(form[$"input_{i}"], out inputValues[i]))
                {
                    return BadRequest("Invalid input value");
                }
            }

            // Score the inputs
            float[] outputValues = _scorer.Score(inputValues);

            // Return the output as a JSON object
            var output = new { values = outputValues };
            return Json(output);
        }
    }
}
