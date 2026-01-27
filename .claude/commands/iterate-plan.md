# Iterate Implementation Plan

You are tasked with updating existing implementation plans based on user feedback. Be skeptical, thorough, and ensure changes are grounded in actual codebase reality.

## Initial Response

When this command is invoked:

1. **Parse the input to identify**:
   - Plan file path (e.g., `docs/plans/2026-01-27-feature.md`)
   - Requested changes/feedback

2. **Handle different scenarios**:

   **If NO plan file provided**:
   ```
   I'll help you iterate on an existing implementation plan.

   Which plan would you like to update? Please provide the path.

   Tip: List recent plans with `ls -lt docs/plans/ | head`
   ```
   Wait for user input.

   **If plan file provided but NO feedback**:
   ```
   I've found the plan at [path]. What changes would you like to make?

   For example:
   - "Add a phase for data augmentation"
   - "Update the success criteria to include GPU memory tests"
   - "Adjust the scope to exclude feature X"
   - "Split Phase 2 into two separate phases"
   ```
   Wait for user input.

   **If BOTH plan file AND feedback provided**:
   Proceed immediately to Step 1.

## Process Steps

### Step 1: Read and Understand Current Plan

1. **Read the existing plan file COMPLETELY**:
   - Use the Read tool WITHOUT limit/offset parameters
   - Understand the current structure, phases, and scope
   - Note the success criteria and implementation approach

2. **Understand the requested changes**:
   - Parse what the user wants to add/modify/remove
   - Identify if changes require codebase research
   - Determine scope of the update

### Step 2: Research If Needed

**Only research if the changes require new technical understanding.**

If the feedback requires understanding new code patterns:

1. **Use Task tool with Explore agent** to research:
   - Find relevant files
   - Understand implementation details
   - Find similar patterns

2. **Read any new files identified** FULLY

3. **Wait for research to complete** before proceeding

### Step 3: Present Understanding and Approach

Before making changes, confirm your understanding:

```
Based on your feedback, I understand you want to:
- [Change 1 with specific detail]
- [Change 2 with specific detail]

My research found:
- [Relevant code pattern or constraint]
- [Important discovery that affects the change]

I plan to update the plan by:
1. [Specific modification to make]
2. [Another modification]

Does this align with your intent?
```

Get user confirmation before proceeding.

### Step 4: Update the Plan

1. **Make focused, precise edits**:
   - Use the Edit tool for surgical changes
   - Maintain the existing structure unless explicitly changing it
   - Keep all file:line references accurate
   - Update success criteria if needed

2. **Ensure consistency**:
   - If adding a new phase, follow the existing pattern
   - If modifying scope, update "What We're NOT Doing" section
   - Maintain automated vs manual success criteria distinction

3. **Preserve quality standards**:
   - Include specific file paths and line numbers
   - Write measurable success criteria
   - Keep language clear and actionable

### Step 5: Review

**Present the changes made**:
```
I've updated the plan at `docs/plans/[filename].md`

Changes made:
- [Specific change 1]
- [Specific change 2]

Would you like any further adjustments?
```

**Be ready to iterate further** based on feedback.

## Important Guidelines

1. **Be Skeptical**:
   - Don't blindly accept problematic change requests
   - Question vague feedback - ask for clarification
   - Verify technical feasibility with code research
   - Point out potential conflicts with existing phases

2. **Be Surgical**:
   - Make precise edits, not wholesale rewrites
   - Preserve good content that doesn't need changing
   - Only research what's necessary
   - Don't over-engineer the updates

3. **Be Thorough**:
   - Read the entire existing plan before making changes
   - Research code patterns if changes require new understanding
   - Ensure updated sections maintain quality standards

4. **Be Interactive**:
   - Confirm understanding before making changes
   - Show what you plan to change before doing it
   - Allow course corrections

5. **No Open Questions**:
   - If the requested change raises questions, ASK
   - Do NOT update the plan with unresolved questions
   - Every change must be complete and actionable

## Success Criteria Guidelines

When updating success criteria, maintain two categories:

1. **Automated Verification**:
   - `uv run task lint`
   - `uv run task test`
   - Specific commands that verify functionality

2. **Manual Verification**:
   - Visual inspection
   - Performance under real conditions
   - Edge cases that are hard to automate
