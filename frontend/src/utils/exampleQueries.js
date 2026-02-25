export const exampleQueries = [
  {
    category: 'netbrain',
    label: 'NetBrain',
    icon: '\uD83C\uDF10',
    queries: [
      'Find path from 10.0.0.1 to 10.0.1.1',
      'Show network path from 192.168.1.1 to 192.168.2.1',
      'Is path allowed from 10.0.0.1 to 10.0.1.1?',
      'Check if traffic from 10.0.0.1 to 10.0.1.1 on TCP 443 is allowed',
      'What is the path between 11.0.0.1 and 11.0.0.250?',
    ],
  },
  {
    category: 'panorama',
    label: 'Panorama',
    icon: '\uD83D\uDD25',
    queries: [
      'What address group is 11.0.0.1 part of?',
      'What IPs are in address group leander_web?',
      'Give me all orphaned address objects',
      'Show unused address groups',
    ],
  },
  {
    category: 'splunk',
    label: 'Splunk',
    icon: '\uD83D\uDCCA',
    queries: [
      'Recent deny events for 10.0.0.250',
    ],
  },
  {
    category: 'docs',
    label: 'Documentation',
    icon: '\uD83D\uDCDA',
    queries: [
      'How does authentication work?',
      'How is tool access enforced?',
      'How do I add a new tool?',
      'How do sessions work?',
    ],
  },
]
