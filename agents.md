I am making an an app to help engineers at oil companies ask questions about their SoPs and P&IDs. Here's the goal: to create a knowledge graph (paired with an entity index) that allows an LLM to answer a users query by traversing the knowledge graph to find information. Additionally, by using the knowledge graph, the LLM should be able to find discrepencies between the details in the SoP and the details in the knowledge graph. Here is the reference problem statement: 

"Input:

A P&ID document at data/pid/diagram.pdf.
An SOP document at data/sop/sop.docx.
Functionality:

P&ID to Graph Conversion:

Process a P&ID image to extract key components (e.g., valves, pumps, sensors) and their connections (edges).
Represent the extracted information as a graph where nodes represent components and edges represent the interconnections.
Capture relevant attributes for each component (e.g., labels, types, specifications).
SOP Cross-Referencing:

Parse a provided SOP document to extract process steps, required components, or other pertinent details.
Compare the graph derived from the P&ID with the SOP details.
Identify and report any inconsistencies between the diagram and the SOP (e.g., missing components, mismatches in connections, or attribute differences).
Output:

A structured graph representation (e.g., using a format compatible with NetworkX or a similar library) that includes nodes with attributes and edges representing connections.
A report or log that details the cross-referencing results, highlighting any discrepancies between the P&ID and the SOP."

## Approach

I want to break this up into two sections of functionaility: Entity & Relation extraction, and then querying. Additionally, we need to have a function that uses the knowledge graph to parse the SoP and determines any differences.

### Entity and Relation Extraction

I want to do this through an iterative approach using a Multi-modal LLM (gpt 5.2). The input is a PDF with 3 pages, each page has a part on it. The LLM should analyze each image, and output a json that describes all entities and all relationships. It can do this one page at a time. The document is extremely compact and detailed, with pressure and temperature measurements everywhere. It's very important that we get AT LEAST the following information wherever its available:

1. Pressure and temperature details for any Part, valve, pipe, or anything else. There will be multiple per page.
2. Connections. Pipes connect parts to different parts, and to different valves, etc. Those connections MUST be documented in the knowledge graph. Knowing how everything connects is super important.
3. How the different pages in the PDF connect to each other (if at all)

I think the best way to do this is the following: Provide the LLM with a tool that allows it to zoom in on the image to get better detail. So prompt gpt 5.2 with a prompt that provides it necessary context, gives it the full image of the part, and provide it with the ability to "request" a bunch more views. So after its first look, it may want to zoom in on the middle where there's a lot of small text, and the right side where there's some text. It can request those 2 views (by providing a list of bounding boxes), and then the tool call will provide it the LLM. Then it can ask more questions if it wants, with a max of 5 loops. 

This should connect to a neo4j database, the information for the DB is in the .env file. Also, it should not use a router, so I think that means you have to use a "bolt" url.

After extracting all entities, we want to make an Entity Index (which a text file about all the entities), so that we can perform rag over the entities to find the right one, before the LLM queries the KG. this is so spelling errors don't arise, like "V-757" instead of "V_757". For each entity, store something like: {
    name: ...
    doc_id: ...
    context: ...
}. Before looking at the KG, provide the LLM with the results of a RAG + Lexical search across this entity index.

### Querying 

Querying should be the easy part. It should be done like this:

1. Use RAG and Lexical search across the entity index to find relevant entities. 
2. Provide the result of this search to an LLM. 
3. The LLM should have access to a tool called "search_knowledge_graph", that takes a Cypher query as input and returns the output of the cypher query. The LLM should be able to use this tool a maximum of 5 times before it gives up.

There should be a separate function to parse discrepencies from the SoP. It should be similar to querying the LLM though, find a way to provide the SoP as well as the KG, and then utilize the KG for information.