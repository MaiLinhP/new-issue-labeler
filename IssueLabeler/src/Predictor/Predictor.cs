// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Actions.Core.Extensions;
using Actions.Core.Services;
using Actions.Core.Summaries;
using GitHubClient;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

using var provider = new ServiceCollection()
    .AddGitHubActionsCore()
    .BuildServiceProvider();

var action = provider.GetRequiredService<ICoreService>();
if (Args.Parse(args, action) is not Args argsData) return 1;

List<Task<(ulong Number, string ResultMessage, bool Success)>> tasks = new();

// Process Category Issues Model
if (argsData.CategoryIssuesModelPath is not null && argsData.Issues is not null)
{
    await action.WriteStatusAsync($"Loading prediction engine for category issues model...");
    var categoryIssueContext = new MLContext();
    var categoryIssueModel = categoryIssueContext.Model.Load(argsData.CategoryIssuesModelPath, out _);
    var categoryIssuePredictor = categoryIssueContext.Model.CreatePredictionEngine<Issue, LabelPrediction>(categoryIssueModel);
    await action.WriteStatusAsync($"Category issues prediction engine ready.");

    foreach (ulong issueNumber in argsData.Issues)
    {
        var result = await GitHubApi.GetIssue(argsData.GitHubToken, argsData.Org, argsData.Repo, issueNumber, argsData.Retries, action, argsData.Verbose);

        if (result is null)
        {
            action.WriteNotice($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] could not be found or downloaded. Skipped.");
            continue;
        }

        if (argsData.ExcludedAuthors is not null && result.Author?.Login is not null && argsData.ExcludedAuthors.Contains(result.Author.Login, StringComparer.InvariantCultureIgnoreCase))
        {
            action.WriteNotice($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] Author '{result.Author.Login}' is in excluded list. Skipped.");
            continue;
        }

        tasks.Add(Task.Run(() => ProcessPrediction(
            categoryIssuePredictor,
            issueNumber,
            new Issue(result),
            argsData.DefaultLabel,
            ModelType.Issue,
            argsData.Retries,
            argsData.Test,
            "Category"
        )));

        action.WriteInfo($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] Queued for category prediction.");
    }
}

// Process Service Issues Model
if (argsData.ServiceIssuesModelPath is not null && argsData.Issues is not null)
{
    await action.WriteStatusAsync($"Loading prediction engine for service issues model...");
    var serviceIssueContext = new MLContext();
    var serviceIssueModel = serviceIssueContext.Model.Load(argsData.ServiceIssuesModelPath, out _);
    var serviceIssuePredictor = serviceIssueContext.Model.CreatePredictionEngine<Issue, LabelPrediction>(serviceIssueModel);
    await action.WriteStatusAsync($"Service issues prediction engine ready.");

    foreach (ulong issueNumber in argsData.Issues)
    {
        var result = await GitHubApi.GetIssue(argsData.GitHubToken, argsData.Org, argsData.Repo, issueNumber, argsData.Retries, action, argsData.Verbose);

        if (result is null)
        {
            action.WriteNotice($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] could not be found or downloaded. Skipped.");
            continue;
        }

        if (argsData.ExcludedAuthors is not null && result.Author?.Login is not null && argsData.ExcludedAuthors.Contains(result.Author.Login, StringComparer.InvariantCultureIgnoreCase))
        {
            action.WriteNotice($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] Author '{result.Author.Login}' is in excluded list. Skipped.");
            continue;
        }

        tasks.Add(Task.Run(() => ProcessPrediction(
            serviceIssuePredictor,
            issueNumber,
            new Issue(result),
            argsData.DefaultLabel,
            ModelType.Issue,
            argsData.Retries,
            argsData.Test,
            "Service"
        )));

        action.WriteInfo($"[Issue {argsData.Org}/{argsData.Repo}#{issueNumber}] Queued for service prediction.");
    }
}

// Process Category Pulls Model
if (argsData.CategoryPullsModelPath is not null && argsData.Pulls is not null)
{
    await action.WriteStatusAsync($"Loading prediction engine for category pulls model...");
    var categoryPullContext = new MLContext();
    var categoryPullModel = categoryPullContext.Model.Load(argsData.CategoryPullsModelPath, out _);
    var categoryPullPredictor = categoryPullContext.Model.CreatePredictionEngine<PullRequest, LabelPrediction>(categoryPullModel);
    await action.WriteStatusAsync($"Category pulls prediction engine ready.");

    foreach (ulong pullNumber in argsData.Pulls)
    {
        var result = await GitHubApi.GetPullRequest(argsData.GitHubToken, argsData.Org, argsData.Repo, pullNumber, argsData.Retries, action, argsData.Verbose);

        if (result is null)
        {
            action.WriteNotice($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] could not be found or downloaded. Skipped.");
            continue;
        }

        if (argsData.ExcludedAuthors is not null && result.Author?.Login is not null && argsData.ExcludedAuthors.Contains(result.Author.Login))
        {
            action.WriteNotice($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] Author '{result.Author.Login}' is in excluded list. Skipped.");
            continue;
        }

        tasks.Add(Task.Run(() => ProcessPrediction(
            categoryPullPredictor,
            pullNumber,
            new PullRequest(result),
            argsData.DefaultLabel,
            ModelType.PullRequest,
            argsData.Retries,
            argsData.Test,
            "Category"
        )));

        action.WriteInfo($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] Queued for category prediction.");
    }
}

// Process Service Pulls Model
if (argsData.ServicePullsModelPath is not null && argsData.Pulls is not null)
{
    await action.WriteStatusAsync($"Loading prediction engine for service pulls model...");
    var servicePullContext = new MLContext();
    var servicePullModel = servicePullContext.Model.Load(argsData.ServicePullsModelPath, out _);
    var servicePullPredictor = servicePullContext.Model.CreatePredictionEngine<PullRequest, LabelPrediction>(servicePullModel);
    await action.WriteStatusAsync($"Service pulls prediction engine ready.");

    foreach (ulong pullNumber in argsData.Pulls)
    {
        var result = await GitHubApi.GetPullRequest(argsData.GitHubToken, argsData.Org, argsData.Repo, pullNumber, argsData.Retries, action, argsData.Verbose);

        if (result is null)
        {
            action.WriteNotice($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] could not be found or downloaded. Skipped.");
            continue;
        }

        if (argsData.ExcludedAuthors is not null && result.Author?.Login is not null && argsData.ExcludedAuthors.Contains(result.Author.Login))
        {
            action.WriteNotice($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] Author '{result.Author.Login}' is in excluded list. Skipped.");
            continue;
        }

        tasks.Add(Task.Run(() => ProcessPrediction(
            servicePullPredictor,
            pullNumber,
            new PullRequest(result),
            argsData.DefaultLabel,
            ModelType.PullRequest,
            argsData.Retries,
            argsData.Test,
            "Service"
        )));

        action.WriteInfo($"[Pull Request {argsData.Org}/{argsData.Repo}#{pullNumber}] Queued for service prediction.");
    }
}

var (predictionResults, success) = await App.RunTasks(tasks, action);

foreach (var prediction in predictionResults.OrderBy(p => p.Number))
{
    action.WriteInfo(prediction.ResultMessage);
}

await action.Summary.WritePersistentAsync();
return success ? 0 : 1;

async Task<(ulong Number, string ResultMessage, bool Success)> ProcessPrediction<T>(PredictionEngine<T, LabelPrediction> predictor, ulong number, T issueOrPull, string? defaultLabel, ModelType type, int[] retries, bool test, string modelName = "") where T : Issue
{
    List<Action<Summary>> predictionResults = [];
    string typeName = type == ModelType.PullRequest ? $"Pull Request ({modelName})" : $"Issue ({modelName})";
    List<string> resultMessageParts = [];
    string? error = null;

    (ulong, string, bool) GetResult(bool success)
    {
        foreach (var summaryWrite in predictionResults)
        {
            action.Summary.AddPersistent(summaryWrite);
        }

        return (number, $"[{typeName} {argsData.Org}/{argsData.Repo}#{number}] {string.Join(' ', resultMessageParts)}", success);
    }

    (ulong, string, bool) Success() => GetResult(true);
    (ulong, string, bool) Failure() => GetResult(false);

    predictionResults.Add(summary => summary.AddRawMarkdown($"- **{argsData.Org}/{argsData.Repo}#{number}**", true));

    if (issueOrPull.HasMoreLabels)
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"    - Skipping prediction. Too many labels applied already; cannot be sure no applicable label is already applied.", true));
        resultMessageParts.Add("Too many labels applied already.");

        return Success();
    }

    // Check if issue already has any labels (might skip prediction if no default label management needed)
    bool hasExistingLabels = issueOrPull.Labels?.Any() ?? false;

    bool hasDefaultLabel =
        (defaultLabel is not null) &&
        (issueOrPull.Labels?.Any(l => l.Equals(defaultLabel, StringComparison.OrdinalIgnoreCase)) ?? false);

    // Skip prediction if there are existing labels and no default label to manage
    if (hasExistingLabels && defaultLabel is null)
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"    - Skipping prediction. Issue already has labels and no default label specified.", true));
        resultMessageParts.Add("Issue already has labels.");
        return Success();
    }

    var prediction = predictor.Predict(issueOrPull);

    if (prediction.Score is null || prediction.Score.Length == 0)
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"    - No prediction was made. The prediction engine did not return any possible predictions.", true));
        resultMessageParts.Add("No prediction was made. The prediction engine did not return any possible predictions.");
        return Success();
    }

    VBuffer<ReadOnlyMemory<char>> labels = default;
    predictor.OutputSchema[nameof(LabelPrediction.Score)].GetSlotNames(ref labels);

    var predictions = prediction.Score
        .Select((score, index) => new
        {
            Score = score,
            Label = labels.GetItemOrDefault(index).ToString()
        })
        // Capture the top 3 for including in the output
        .OrderByDescending(p => p.Score)
        .Take(3);

    var bestScore = predictions.FirstOrDefault(p => p.Score >= argsData.Threshold);

    if (bestScore is not null)
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"    - Predicted label: `{bestScore.Label}` meets the threshold of {argsData.Threshold}.", true));
    }
    else
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"    - No label prediction met the threshold of {argsData.Threshold}.", true));
    }

    foreach (var labelPrediction in predictions)
    {
        predictionResults.Add(summary => summary.AddRawMarkdown($"        - `{labelPrediction.Label}` - Score: {labelPrediction.Score}", true));
    }

    if (bestScore is not null)
    {
        if (!test)
        {
            error = await GitHubApi.AddLabel(argsData.GitHubToken, argsData.Org, argsData.Repo, typeName, number, bestScore.Label, retries, action);
        }

        if (error is null)
        {
            predictionResults.Add(summary => summary.AddRawMarkdown($"    - **`{bestScore.Label}` applied**", true));
            resultMessageParts.Add($"Label '{bestScore.Label}' applied.");

            if (hasDefaultLabel && defaultLabel is not null)
            {
                if (!test)
                {
                    error = await GitHubApi.RemoveLabel(argsData.GitHubToken, argsData.Org, argsData.Repo, typeName, number, defaultLabel, retries, action);
                }

                if (error is null)
                {
                    predictionResults.Add(summary => summary.AddRawMarkdown($"    - **Removed default label `{defaultLabel}`**", true));
                    resultMessageParts.Add($"Default label '{defaultLabel}' removed.");
                    return Success();
                }
                else
                {
                    predictionResults.Add(summary => summary.AddRawMarkdown($"    - **Error removing default label `{defaultLabel}`**: {error}", true));
                    resultMessageParts.Add($"Error occurred removing default label '{defaultLabel}'");
                    return Failure();
                }
            }
            else
            {
                return Success();
            }
        }
        else
        {
            predictionResults.Add(summary => summary.AddRawMarkdown($"    - **Error applying label `{bestScore.Label}`**: {error}", true));
            resultMessageParts.Add($"Error occurred applying label '{bestScore.Label}'");
            return Failure();
        }
    }

    if (defaultLabel is not null)
    {
        if (hasDefaultLabel)
        {
            predictionResults.Add(summary => summary.AddRawMarkdown($"    - Default label `{defaultLabel}` is already applied.", true));
            resultMessageParts.Add($"No prediction made. Default label '{defaultLabel}' is already applied.");
            return Success();
        }
        else
        {
            if (!test)
            {
                error = await GitHubApi.AddLabel(argsData.GitHubToken, argsData.Org, argsData.Repo, typeName, number, defaultLabel, argsData.Retries, action);
            }

            if (error is null)
            {
                predictionResults.Add(summary => summary.AddRawMarkdown($"    - **Default label `{defaultLabel}` applied.**", true));
                resultMessageParts.Add($"No prediction made. Default label '{defaultLabel}' applied.");
                return Success();
            }
            else
            {
                predictionResults.Add(summary => summary.AddRawMarkdown($"    - **Error applying default label `{defaultLabel}`**: {error}", true));
                resultMessageParts.Add($"Error occurred applying default label '{defaultLabel}'");
                return Failure();
            }
        }
    }

    resultMessageParts.Add("No prediction made. No applicable label found. No action taken.");
    return GetResult(error is null);
}
