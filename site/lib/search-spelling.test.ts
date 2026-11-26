import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { staticClient } from "fumadocs-core/search/client/orama-static";
import { createSearchAPI } from "fumadocs-core/search/server";
import {
  expandHighlightQuery,
  normalizeSearchQuery,
  normalizeSpellingToken,
} from "./search-spelling.ts";

describe("normalizeSpellingToken", () => {
  it("normalizes common UK spellings and inflections to US English", () => {
    const cases: ReadonlyArray<readonly [string, string]> = [
      // -re -> -er
      ["centre", "center"],
      ["centres", "centers"],
      ["centred", "centered"],
      ["centring", "centering"],
      ["isocentre", "isocenter"],
      ["recentre", "recenter"],
      ["recentred", "recentered"],
      ["metre", "meter"],
      ["metres", "meters"],
      ["millimetre", "millimeter"],
      ["millimetres", "millimeters"],
      ["micrometre", "micrometer"],
      ["micrometres", "micrometers"],
      ["nanometre", "nanometer"],
      ["centimetre", "centimeter"],
      ["kilometre", "kilometer"],
      ["fibre", "fiber"],
      ["fibres", "fibers"],
      ["litre", "liter"],
      ["millilitre", "milliliter"],
      ["calibre", "caliber"],

      // compounds and vendor/schema identifiers
      ["detector-centre", "detector-center"],
      ["centre_sweep", "center_sweep"],
      ["CentreOfRotationTop", "centerofrotationtop"],
      ["CentreOfRotationBottom", "centerofrotationbottom"],
      ["AutomaticCentreOfRotation", "automaticcenterofrotation"],
      ["CentreSweep", "centersweep"],
      ["CentrePlan", "centerplan"],
      ["CentreSelection", "centerselection"],
      ["SimulationArtefacts", "simulationartifacts"],
      ["artefactmetadata", "artifactmetadata"],
      ["batchlinearisation", "batchlinearization"],
      ["deapodised", "deapodized"],
      ["tv-regularised", "tv-regularized"],
      ["nearest-neighbour", "nearest-neighbor"],
      ["feature_colours", "feature_colors"],
      ["open_neighbours", "open_neighbors"],
      ["centre's", "center's"],

      // -is* -> -iz*
      ["regularise", "regularize"],
      ["regularised", "regularized"],
      ["regularises", "regularizes"],
      ["regulariser", "regularizer"],
      ["regularisers", "regularizers"],
      ["regularising", "regularizing"],
      ["regularisation", "regularization"],
      ["unregularised", "unregularized"],
      ["optimise", "optimize"],
      ["optimised", "optimized"],
      ["optimiser", "optimizer"],
      ["optimising", "optimizing"],
      ["optimisation", "optimization"],
      ["normalise", "normalize"],
      ["normalised", "normalized"],
      ["normalisers", "normalizers"],
      ["normalisation", "normalization"],
      ["unnormalised", "unnormalized"],
      ["initialise", "initialize"],
      ["initialised", "initialized"],
      ["initialiser", "initializer"],
      ["initialisation", "initialization"],
      ["voxelised", "voxelized"],
      ["linearise", "linearize"],
      ["linearisation", "linearization"],
      ["minimise", "minimize"],
      ["minimising", "minimizing"],
      ["maximise", "maximize"],
      ["maximising", "maximizing"],
      ["discretise", "discretize"],
      ["discretisation", "discretization"],
      ["serialise", "serialize"],
      ["serialised", "serialized"],
      ["serialisable", "serializable"],
      ["visualise", "visualize"],
      ["visualisation", "visualization"],
      ["recognise", "recognize"],
      ["recognised", "recognized"],
      ["unrecognised", "unrecognized"],
      ["summarise", "summarize"],
      ["summarised", "summarized"],
      ["vectorise", "vectorize"],
      ["vectorised", "vectorized"],
      ["randomise", "randomize"],
      ["randomised", "randomized"],
      ["parameterise", "parameterize"],
      ["parameterisation", "parameterization"],
      ["parametrise", "parameterize"],
      ["parametrisation", "parameterization"],
      ["parametrize", "parameterize"],
      ["materialise", "materialize"],
      ["materialisation", "materialization"],
      ["externalise", "externalize"],
      ["externalised", "externalized"],
      ["canonicalise", "canonicalize"],
      ["contextualise", "contextualize"],
      ["penalised", "penalized"],
      ["rasterisation", "rasterization"],
      ["synthesised", "synthesized"],
      ["realise", "realize"],
      ["emphasise", "emphasize"],
      ["analyse", "analyze"],
      ["analysed", "analyzed"],
      ["analysing", "analyzing"],
      ["analyser", "analyzer"],

      // -our -> -or
      ["colour", "color"],
      ["colours", "colors"],
      ["coloured", "colored"],
      ["colouring", "coloring"],
      ["colourmap", "colormap"],
      ["colourbar", "colorbar"],
      ["behaviour", "behavior"],
      ["behaviours", "behaviors"],
      ["behavioural", "behavioral"],
      ["neighbour", "neighbor"],
      ["neighbours", "neighbors"],
      ["neighbouring", "neighboring"],
      ["neighbourhood", "neighborhood"],
      ["honour", "honor"],
      ["favour", "favor"],
      ["favours", "favors"],
      ["favourable", "favorable"],
      ["favourite", "favorite"],

      // doubled l -> single l
      ["labelled", "labeled"],
      ["labelling", "labeling"],
      ["unlabelled", "unlabeled"],
      ["relabelling", "relabeling"],
      ["modelled", "modeled"],
      ["modelling", "modeling"],
      ["unmodelled", "unmodeled"],
      ["cancelled", "canceled"],
      ["cancelling", "canceling"],
      ["travelling", "traveling"],
      ["signalled", "signaled"],
      ["levelling", "leveling"],
      ["channelling", "channeling"],

      // other UK/US pairs
      ["artefact", "artifact"],
      ["artefacts", "artifacts"],
      ["grey", "gray"],
      ["greys", "grays"],
      ["greyscale", "grayscale"],
      ["aluminium", "aluminum"],
      ["licence", "license"],
      ["licences", "licenses"],
      ["licenced", "licensed"],
      ["licencing", "licensing"],
      ["analogue", "analog"],
      ["analogues", "analogs"],
      ["catalogue", "catalog"],
      ["catalogues", "catalogs"],
      ["catalogued", "cataloged"],
      ["cataloguing", "cataloging"],
      ["dialogue", "dialog"],
      ["anticlockwise", "counterclockwise"],
      ["anti-clockwise", "counterclockwise"],
      ["acknowledgement", "acknowledgment"],
      ["judgement", "judgment"],
      ["zeroes", "zeros"],
      ["defence", "defense"],
      ["programme", "program"],
      ["practise", "practice"],
      ["sulphur", "sulfur"],
      ["caesium", "cesium"],
      ["ageing", "aging"],
      ["manoeuvre", "maneuver"],
      ["manoeuvred", "maneuvered"],
      ["manoeuvring", "maneuvering"],
    ];

    for (const [input, expected] of cases) {
      assert.equal(
        normalizeSpellingToken(input),
        expected,
        `expected "${input}" -> "${expected}"`,
      );
    }
  });

  it("handles incremental typing prefixes cleanly", () => {
    const prefixes: ReadonlyArray<readonly [string, string]> = [
      ["regularis", "regulariz"],
      ["regularisa", "regulariza"],
      ["regularisat", "regularizat"],
      ["regularisati", "regularizati"],
      ["regularisatio", "regularizatio"],
      ["regularisin", "regularizin"],
      ["optimis", "optimiz"],
      ["normalis", "normaliz"],
      ["initialis", "initializ"],
      ["discretis", "discretiz"],
      ["serialis", "serializ"],
      ["serialisab", "serializab"],
      ["voxelis", "voxeliz"],
      ["linearis", "lineariz"],
      ["parametris", "parameteriz"],
      ["parameteris", "parameteriz"],
      ["artef", "artif"],
      ["artefa", "artifa"],
      ["artefac", "artifac"],
      ["colou", "color"],
      ["behaviou", "behavior"],
      ["neighbou", "neighbor"],
      ["honou", "honor"],
      ["favou", "favor"],
      ["labell", "label"],
      ["unlabell", "unlabel"],
      ["modell", "model"],
      ["unmodell", "unmodel"],
      ["cancell", "cancel"],
      ["cancelle", "cancele"],
      ["cancelli", "canceli"],
      ["centrin", "centerin"],
      ["licenc", "licens"],
      ["analogu", "analog"],
      ["catalogu", "catalog"],
      ["alumini", "aluminu"],
      ["aluminiu", "aluminu"],
      ["anticlock", "counterclock"],
    ];

    for (const [input, expected] of prefixes) {
      assert.equal(
        normalizeSpellingToken(input),
        expected,
        `expected prefix "${input}" -> "${expected}"`,
      );
    }
  });

  it("does not mangle valid non-UK words", () => {
    const unchanged = [
      "central",
      "centric",
      "centroid",
      "centrifuge",
      "concentrate",
      "descentrecord",
      "realistic",
      "realism",
      "realist",
      "realis",
      "optimism",
      "optimist",
      "optimistic",
      "denoiser",
      "noise",
      "pairwise",
      "elementwise",
      "bitwise",
      "precise",
      "concise",
      "emphasis",
      "synthesis",
      "hypothesis",
      "analysis",
      "analyses",
      "analytic",
      "alumina",
      "analogous",
      "cancellation",
      "metric",
      "metrics",
      "geometry",
      "symmetry",
      "diameter",
      "parameter",
      "parametric",
      "spectrometer",
      "artery",
      "arterial",
      "gre",
      "green",
      "greater",
      "our",
      "your",
      "four",
      "hour",
      "contour",
      "source",
      "course",
      "fourier",
      "advertise",
      "exercise",
      "promise",
      "raise",
      "revise",
      "rise",
      "kaiser",
      "disable",
      "madvise",
      "sizing",
      "resizing",
      "mechanism",
      "organism",
      "criticism",
      "formalism",
      "minimalist",
      "specialist",
      "programmed",
      "programming",
      "fibrous",
      "literal",
      "calibrate",
    ];

    for (const word of unchanged) {
      assert.equal(
        normalizeSpellingToken(word),
        word,
        `expected "${word}" to remain unchanged`,
      );
    }
  });
});

describe("normalizeSearchQuery and expandHighlightQuery", () => {
  it("normalizes multi-word queries while preserving separators", () => {
    assert.equal(normalizeSearchQuery("detector centre"), "detector center");
    assert.equal(normalizeSearchQuery("centre sweep"), "center sweep");
    assert.equal(
      normalizeSearchQuery("FISTA-TV regularisation & ring artefacts"),
      "fista-tv regularization & ring artifacts",
    );
    assert.equal(normalizeSearchQuery("'centre'"), "'center'");
    assert.equal(normalizeSearchQuery("anti-clockwise"), "counterclockwise");
  });

  it("expands highlight queries symmetrically without duplicating unchanged terms", () => {
    assert.equal(expandHighlightQuery("centre"), "centre center");
    assert.equal(expandHighlightQuery("center"), "center centre");
    assert.equal(
      expandHighlightQuery("detector centre"),
      "detector centre center",
    );
    assert.equal(
      expandHighlightQuery("regulariser"),
      "regulariser regularizer",
    );
    assert.equal(
      expandHighlightQuery("regularizer"),
      "regularizer regulariser",
    );
    assert.equal(
      expandHighlightQuery("CenterOfRotationTop"),
      "CenterOfRotationTop centreofrotationtop",
    );
    assert.equal(expandHighlightQuery("fista"), "fista");
    assert.equal(expandHighlightQuery("central"), "central");
    assert.equal(expandHighlightQuery(""), "");
  });
});

async function withMockedStaticIndex<T>(
  url: string,
  payload: unknown,
  fn: () => Promise<T>,
): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request) => {
    const reqUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    if (reqUrl === url) {
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    throw new Error(`Unexpected fetch in test: ${reqUrl}`);
  }) as typeof fetch;

  try {
    return await fn();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

describe("Fumadocs static search pipeline integration", () => {
  it("matches UK queries against US docs and US queries against UK vendor keys", async () => {
    const server = createSearchAPI("advanced", {
      tokenizer: {
        language: "english",
        stemmer: normalizeSpellingToken,
      },
      indexes: [
        {
          id: "/align-a-scan/",
          url: "/align-a-scan/",
          title: "Align a scan",
          description:
            "Find the detector center, run center_sweep, and optimize geometry.",
          structuredData: {
            headings: [{ id: "detector-center", content: "Detector center" }],
            contents: [
              {
                heading: "detector-center",
                content:
                  "Use detector-center calibration, center_sweep, and Nikon CentreOfRotationTop and AutomaticCentreOfRotation metadata. Counterclockwise rotation is positive.",
              },
            ],
          },
        },
        {
          id: "/reconstruct/",
          url: "/reconstruct/",
          title: "Reconstruction and regularization",
          description:
            "FISTA-TV regularizer, normalized projections, and ring artifact removal.",
          structuredData: {
            headings: [{ id: "regularization", content: "Regularization" }],
            contents: [
              {
                heading: "regularization",
                content:
                  "Tune the regularizer to suppress artifacts, inspect grayscale and color slices, compare neighbors, and handle canceled jobs, labeled panels, and modeled motion.",
              },
              {
                heading: "regularization",
                content:
                  "Voxelized phantoms use millimeter and micrometer spacing, aluminum filters, linearized updates, initialized poses, minimized objectives, maximized contrast, discretized operators, serialized state, visualized slices, recognized formats, summarized metrics, vectorized kernels, randomized splits, parameterized curves, materialized buffers, and externalized storage.",
              },
              {
                heading: "regularization",
                content:
                  "Expected behavior honors defaults, favors stability, ships under an open license, supports analog sensors, and reads a scan catalog.",
              },
            ],
          },
        },
        {
          id: "/guards/",
          url: "/guards/",
          title: "Non-UK guard terms",
          description: "Central slice, denoiser, alumina, realistic noise.",
          structuredData: {
            headings: [],
            contents: [
              {
                heading: undefined,
                content:
                  "Central slice extraction, realistic phantom noise, denoiser comparisons, alumina sample, pairwise emphasis, precise cancellation checks.",
              },
            ],
          },
        },
      ],
    });

    const exported = await server.export();
    const url = "https://tomojax.test/api/search-integration";

    await withMockedStaticIndex(url, exported, async () => {
      const client = staticClient({ from: url });
      const search = (query: string) =>
        client.search(normalizeSearchQuery(query));

      const expectedHits: ReadonlyArray<readonly [string, string]> = [
        // UK queries -> US docs
        ["centre", "/align-a-scan/"],
        ["detector centre", "/align-a-scan/"],
        ["detector-centre", "/align-a-scan/"],
        ["centre sweep", "/align-a-scan/"],
        ["centre_sweep", "/align-a-scan/"],
        ["optimise", "/align-a-scan/"],
        ["anticlockwise", "/align-a-scan/"],
        ["anti-clockwise", "/align-a-scan/"],
        ["regulariser", "/reconstruct/"],
        ["regularisation", "/reconstruct/"],
        ["regularis", "/reconstruct/"],
        ["artefact", "/reconstruct/"],
        ["artef", "/reconstruct/"],
        ["colour", "/reconstruct/"],
        ["colou", "/reconstruct/"],
        ["greyscale", "/reconstruct/"],
        ["neighbours", "/reconstruct/"],
        ["neighbou", "/reconstruct/"],
        ["normalised", "/reconstruct/"],
        ["cancelled", "/reconstruct/"],
        ["cancell", "/reconstruct/"],
        ["labelled", "/reconstruct/"],
        ["modelled", "/reconstruct/"],
        ["millimetre", "/reconstruct/"],
        ["micrometre", "/reconstruct/"],
        ["aluminium", "/reconstruct/"],
        ["alumini", "/reconstruct/"],
        ["voxelised", "/reconstruct/"],
        ["linearised", "/reconstruct/"],
        ["initialised", "/reconstruct/"],
        ["minimised", "/reconstruct/"],
        ["maximised", "/reconstruct/"],
        ["discretised", "/reconstruct/"],
        ["serialised", "/reconstruct/"],
        ["visualised", "/reconstruct/"],
        ["recognised", "/reconstruct/"],
        ["summarised", "/reconstruct/"],
        ["vectorised", "/reconstruct/"],
        ["randomised", "/reconstruct/"],
        ["parametrised", "/reconstruct/"],
        ["parameterised", "/reconstruct/"],
        ["materialised", "/reconstruct/"],
        ["externalised", "/reconstruct/"],
        ["behaviour", "/reconstruct/"],
        ["honours", "/reconstruct/"],
        ["favours", "/reconstruct/"],
        ["licence", "/reconstruct/"],
        ["analogue", "/reconstruct/"],
        ["catalogue", "/reconstruct/"],

        // US queries -> remaining UK vendor keys
        ["CenterOfRotationTop", "/align-a-scan/"],
        ["AutomaticCenterOfRotation", "/align-a-scan/"],
        ["CentreOfRotationTop", "/align-a-scan/"],

        // Non-UK guards stay searchable
        ["central", "/guards/"],
        ["realistic", "/guards/"],
        ["denoiser", "/guards/"],
        ["alumina", "/guards/"],
        ["emphasis", "/guards/"],
        ["pairwise", "/guards/"],
        ["precise", "/guards/"],
        ["noise", "/guards/"],
        ["cancellation", "/guards/"],
      ];

      for (const [query, expectedUrl] of expectedHits) {
        const results = await search(query);
        assert.ok(
          results.some((r) => r.url.startsWith(expectedUrl)),
          `expected query "${query}" to match "${expectedUrl}", got: ${results.map((r) => r.url).join(", ")}`,
        );
      }

      // Verify stored display text preserves original vendor identifier verbatim.
      const nikonResults = await search("CenterOfRotationTop");
      assert.ok(
        nikonResults.some((r) => r.content.includes("CentreOfRotationTop")),
        "expected stored result text to preserve verbatim CentreOfRotationTop",
      );
    });
  });
});

describe("Generated static search index end-to-end", () => {
  const indexPath = path.resolve(import.meta.dirname, "../out/api/search");

  it("returns real TomoJAX docs hits for UK and US queries", {
    skip: !existsSync(indexPath),
  }, async () => {
    const exported = JSON.parse(readFileSync(indexPath, "utf8"));
    const url = "https://tomojax.test/api/search-built";

    await withMockedStaticIndex(url, exported, async () => {
      const client = staticClient({ from: url });
      const search = (query: string) =>
        client.search(normalizeSearchQuery(query));

      const queries = [
        // UK queries
        "centre",
        "detector centre",
        "centre sweep",
        "regulariser",
        "regularisation",
        "regularis",
        "artefact",
        "colour",
        "greyscale",
        "neighbours",
        "normalised",
        "cancelled",
        "labelled",
        "modelled",
        "millimetre",
        "behaviour",
        "licence",
        "catalogue",
        "initialise",
        "optimise",
        "minimise",
        "voxelised",
        "linearise",
        // US queries
        "center",
        "detector center",
        "regularizer",
        "regularization",
        "artifact",
        "color",
        "grayscale",
        "neighbors",
        "normalized",
        "canceled",
        // Non-UK guards
        "central",
        "noise",
        "alumina",
      ];

      for (const query of queries) {
        const results = await search(query);
        assert.ok(
          results.length > 0,
          `expected built static index to return hits for "${query}"`,
        );
      }
    });
  });
});
